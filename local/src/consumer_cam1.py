import cv2
import numpy as np
import json
import time
import datetime
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils

import config
import gui_utils as gui
from log_utils import setup_logging

log = setup_logging("cam1")

# --- CONFIG ---
BROKER = config.BROKER
TOPIC_VIDEO = config.TOPIC_VIDEO
TOPIC_GALLERY = config.TOPIC_GALLERY
GROUP_ID = config.GROUP_CAM1

# Model Paths
YOLO_MODEL_PATH = config.YOLO_CAM1
OSNET_MODEL_PATH = config.OSNET_WEIGHTS

# --- TUNING ---
SKIP_FRAMES = 2
YOLO_CONF = 0.50
MIN_AREA_THRESHOLD = 800
SEND_INTERVAL = 0.5
TRACK_THRESH = 0.2
MATCH_THRESH = 0.8
DATA_FPS = 30
HEARTBEAT_SEC = 5.0


def run_cam1():
    log.info("Init YOLO", extra={"event": "yolo_init", "weights": YOLO_MODEL_PATH})
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception:
        log.error("Failed to load YOLO, check model path", exc_info=True,
                  extra={"event": "yolo_error", "weights": YOLO_MODEL_PATH})
        return

    res_writer = utils.MOTResultWriter(
        output_path=config.res_path("cam1"),
        target_width=None,  # or 640 if the producer streams at 640
        original_width=1920,
        original_height=1080,
        logger=log,
    )

    # Init Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_THRESH,
        lost_track_buffer=30,
        minimum_matching_threshold=MATCH_THRESH,
        frame_rate=DATA_FPS
    )

    # Init Feature Extractor with a dynamic path
    extractor = utils.FeatureExtractor(model_path=OSNET_MODEL_PATH, logger=log)

    # Init Kafka via utils
    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO])
    producer = utils.get_kafka_producer(BROKER)

    box_an = sv.BoxAnnotator(thickness=2)

    frame_count = 0
    sent_count = 0
    cam1_last_sent = {}  # cache: {track_id: timestamp}
    last_hb = time.time()
    hb_frame_mark = 0

    log.info("Cam1 ready (gallery source)", extra={"event": "ready", "broker": BROKER})

    try:
        while True:
            # Heartbeat first, on wall-clock time: fires even when no frames arrive,
            # so a frame-starved consumer still proves it's alive (fps reads 0) instead
            # of looking dead in the logs.
            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                fps = (frame_count - hb_frame_mark) / (now - last_hb)
                log.info("Cam1 stats", extra={"event": "heartbeat", "fps": round(fps, 1),
                                              "frames": frame_count, "gallery_sent": sent_count,
                                              "tracks": len(cam1_last_sent)})
                last_hb = now
                hb_frame_mark = frame_count

            msg = consumer.poll(0.01)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka message error", extra={"event": "kafka_error", "error": str(msg.error())})
                continue

            # Only process cam1 messages
            if msg.key().decode() != "cam1":
                continue

            nparr = np.frombuffer(msg.value(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_count += 1

            # --- ACCURATE TIMESTAMP ---
            ts_now = time.time()
            kafka_frame_idx = frame_count
            try:
                headers = msg.headers()
                if headers:
                    meta = json.loads(headers[0][1].decode())
                    ts_now = meta.get('timestamp', ts_now)
                    # Producer sends frame_idx from 0, GT starts at 1
                    kafka_frame_idx = meta.get('frame_idx', frame_count) + 1
            except (json.JSONDecodeError, IndexError, AttributeError):
                log.debug("Could not read meta header, using time.time()", extra={"event": "meta_fallback"})

            dt_object = datetime.datetime.fromtimestamp(ts_now)
            time_str = dt_object.strftime('%H:%M:%S.%f')[:-4]

            detections_display = None

            if frame_count % (SKIP_FRAMES + 1) == 0:
                results = model(frame, verbose=False, conf=YOLO_CONF)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Use the helper from utils
                detections = utils.merge_truck_boxes(detections, frame.shape)
                detections = tracker.update_with_detections(detections)
                detections_display = detections

                for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                    res_writer.write(kafka_frame_idx, tid, xyxy, conf)

                # Prepare crops
                crops = []
                valid_indices = []

                for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                    x1, y1, x2, y2 = map(int, xyxy)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    area = (x2 - x1) * (y2 - y1)

                    if area > MIN_AREA_THRESHOLD:
                        crops.append(frame[y1:y2, x1:x2])
                        valid_indices.append(i)

                if crops:
                    # Call extract on the extractor instance
                    embeddings = extractor.extract_batch(crops)

                    for idx, feat_vec in zip(valid_indices, embeddings):
                        tid = int(detections.tracker_id[idx])
                        last_sent_ts = cam1_last_sent.get(tid, 0)

                        if ts_now - last_sent_ts > SEND_INTERVAL:
                            payload = {
                                "track_id": tid,
                                "timestamp": ts_now,
                                "feature": feat_vec.tolist(),
                                "cam_id": "cam1",
                                # Use the encode helper from utils
                                "image_b64": utils.encode_image_base64(crops[valid_indices.index(idx)])
                            }

                            producer.produce(
                                TOPIC_GALLERY,
                                key=str(tid).encode(),
                                value=json.dumps(payload).encode()
                            )
                            cam1_last_sent[tid] = ts_now
                            sent_count += 1
                            log.debug("Gallery sent", extra={"event": "gallery_send", "track_id": tid})

                    producer.poll(0)

            # UI
            if detections_display:
                frame = box_an.annotate(frame, detections_display)

            cv2.putText(frame, f"Cam 1: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            gui.imshow("Cam 1 - Sender", frame)
            if gui.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        log.info("Ctrl-C received, stopping cam1", extra={"event": "shutdown"})
    finally:
        consumer.close()
        res_writer.close()
        gui.destroyAllWindows()
        log.info("Cam1 exited", extra={"event": "exit", "frames": frame_count, "gallery_sent": sent_count})


if __name__ == "__main__":
    run_cam1()
