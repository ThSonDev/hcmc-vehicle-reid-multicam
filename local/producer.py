import cv2
import time
import json
import threading
import signal
import os
from confluent_kafka import Producer

import config
from log_utils import setup_logging

log = setup_logging("producer")

# --- CONFIG ---
BROKER = config.BROKER
TOPIC = config.TOPIC_VIDEO
VIDEO_SOURCES = config.VIDEO_SOURCES

# --- TUNING ---
TARGET_WIDTH = None
TARGET_FPS = 30
JPEG_QUALITY = 70
HEARTBEAT_SEC = 5.0  # per-cam FPS stats interval
MAX_LAG_SEC = 1.0    # if pacing falls this far behind (stall), re-anchor instead of bursting

# Precomputed once (avoid rebuilding the param list per frame)
ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

# REID_REPLAY=0 -> stream each video once then stop (single pass, needed for clean
# GT evaluation via report.py --eval). Default 1 -> loop forever (run.py --once sets 0).
REPLAY = os.environ.get("REID_REPLAY", "1") != "0"

exit_event = threading.Event()


def signal_handler(sig, frame):
    log.info("Stop signal received, exiting...", extra={"event": "shutdown"})
    exit_event.set()


signal.signal(signal.SIGINT, signal_handler)


def delivery_report(err, msg):
    if err is not None:
        log.warning("Frame delivery failed", extra={"event": "delivery_error", "error": str(err)})


def stream_worker(camera_id, video_path, producer, start_time_ref):
    if not os.path.exists(video_path):
        log.error("Video not found", extra={"event": "video_missing", "cam": camera_id, "path": video_path})
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video", extra={"event": "video_open_failed", "cam": camera_id, "path": video_path})
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip_interval = max(1, int(round(original_fps / TARGET_FPS)))
    key = camera_id.encode('utf-8')

    log.info("Stream started", extra={"event": "stream_start", "cam": camera_id,
                                      "target_fps": TARGET_FPS, "path": video_path})

    frame_count = 0
    sent_count = 0
    decoded_this_pass = 0          # guards against a corrupt video looping forever
    epoch = start_time_ref         # pacing anchor (monotonic); re-anchored on replay
    last_hb = time.time()
    hb_sent_mark = 0
    while not exit_event.is_set():
        # grab() advances the decoder without copying/decoding the frame; we only
        # retrieve() the frames we actually send, so dropping FPS saves real CPU.
        if not cap.grab():
            if not REPLAY:
                log.info("End of video, single pass done", extra={"event": "stream_done",
                                                                  "cam": camera_id, "sent_total": sent_count})
                break
            if decoded_this_pass == 0:
                log.error("No frames decoded, aborting worker", extra={"event": "no_frames", "cam": camera_id})
                break
            log.debug("End of video, replaying from start", extra={"event": "replay", "cam": camera_id})
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            decoded_this_pass = 0
            epoch = time.monotonic()   # re-anchor so a looping run stays real-time (no burst)
            continue
        decoded_this_pass += 1

        # Real-time pacing against the per-pass epoch (monotonic = NTP-safe)
        target_ts = frame_count * (1.0 / original_fps)
        wait = target_ts - (time.monotonic() - epoch)
        if wait > 0:
            time.sleep(wait)
        elif wait < -MAX_LAG_SEC:
            # Fell too far behind (stall); re-anchor instead of flooding Kafka to catch up
            epoch = time.monotonic() - target_ts

        if frame_count % frame_skip_interval != 0:
            frame_count += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            log.debug("Frame retrieve failed", extra={"event": "retrieve_failed", "cam": camera_id})
            frame_count += 1
            continue

        if TARGET_WIDTH is not None:
            h, w = frame.shape[:2]
            if w != TARGET_WIDTH:
                scale = TARGET_WIDTH / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ok, buffer = cv2.imencode('.jpg', frame, ENCODE_PARAMS)
        if not ok:
            log.warning("JPEG encode failed", extra={"event": "encode_failed", "cam": camera_id, "frame_idx": frame_count})
            frame_count += 1
            continue

        meta = {"timestamp": time.time(), "frame_idx": frame_count}

        try:
            producer.produce(
                TOPIC,
                key=key,
                value=buffer.tobytes(),
                headers=[("meta", json.dumps(meta).encode('utf-8'))],
                on_delivery=delivery_report
            )
            producer.poll(0)
            sent_count += 1
        except BufferError:
            log.debug("Kafka buffer full, waiting for flush", extra={"event": "buffer_full", "cam": camera_id})
            producer.poll(0.1)

        frame_count += 1

        # Heartbeat: actual send FPS + producer queue depth (backpressure signal)
        now = time.time()
        if now - last_hb >= HEARTBEAT_SEC:
            fps = (sent_count - hb_sent_mark) / (now - last_hb)
            log.info("Stream stats", extra={"event": "heartbeat", "cam": camera_id,
                                            "sent_fps": round(fps, 1), "sent_total": sent_count,
                                            "queue": len(producer)})
            last_hb = now
            hb_sent_mark = sent_count

    cap.release()
    log.info("Worker stopped", extra={"event": "worker_stop", "cam": camera_id, "sent_total": sent_count})


if __name__ == "__main__":
    conf = {
        'bootstrap.servers': BROKER,
        # Steady state is ~90 msg/s with queue depth <=5; this buffer exists only to ride
        # out a transient broker/disk stall (pSLC cache fill). 64 MB ~= a few seconds of
        # frames -> ample headroom, tight memory cap. Raise only if heartbeats show queue growth.
        'queue.buffering.max.messages': 2000,
        'queue.buffering.max.kbytes': 65536,  # 64 MB hard cap on producer memory (frames are large)
        'linger.ms': 10,
        'compression.type': 'lz4'
    }
    producer = Producer(conf)
    threads = []
    global_start = time.monotonic()  # shared pacing anchor -> keeps the 3 cams time-aligned

    log.info("Producer starting", extra={"event": "init", "broker": BROKER,
                                         "topic": TOPIC, "cams": list(VIDEO_SOURCES)})

    for cam_id, path in VIDEO_SOURCES.items():
        t = threading.Thread(target=stream_worker, args=(cam_id, path, producer, global_start))
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        while not exit_event.is_set():
            time.sleep(1)
            if not any(t.is_alive() for t in threads):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Stop workers and let them release their captures before flushing, so the
        # Kafka client tears down cleanly (avoids a destructor race on shutdown).
        exit_event.set()
        for t in threads:
            t.join(timeout=2.0)
        producer.flush(timeout=5.0)
        log.info("Producer exited", extra={"event": "exit"})
