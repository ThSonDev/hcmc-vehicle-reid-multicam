import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils

import config
import gui_utils as gui
from log_utils import setup_logging

log = setup_logging("cam2")

# --- CONFIG ---
BROKER = config.BROKER
TOPIC_VIDEO = config.TOPIC_VIDEO
TOPIC_GALLERY = config.TOPIC_GALLERY
TOPIC_MATCH = config.TOPIC_MATCH
GROUP_ID = config.GROUP_CAM2

YOLO_MODEL = config.YOLO_CAM2
OSNET_MODEL = config.OSNET_WEIGHTS

SIMILARITY_THRESHOLD = 0.65
MIN_TRAVEL_TIME = 1.0
MAX_TRAVEL_TIME = 10
HEARTBEAT_SEC = 5.0

# Long-run state eviction: a gallery entry older than MAX_TRAVEL_TIME can never
# satisfy the travel-time gate again, so it is dead weight (same policy as Cam 3).
# Locks live a bit longer (LOCK_TTL >= MAX_TRAVEL_TIME) so a lock always outlives
# the gallery entry it guards -> no premature re-match -- while still being freed
# on long runs / replays so IDs can be reused instead of leaking forever.
LOCK_TTL = MAX_TRAVEL_TIME


def evict_stale_state(gallery_db, locked_cam1_ids, locked_cam2_map, now):
    """Drop gallery entries and ID locks past their TTL (timestamp-based).

    Mirrors Cam 3's gallery eviction and extends the same idea to the lock
    tables, keeping memory bounded and the per-frame gallery scan small during
    extended / repeated runs."""
    for k in [k for k, v in gallery_db.items() if now - v['ts'] > MAX_TRAVEL_TIME]:
        del gallery_db[k]
    for k in [k for k, ts in locked_cam1_ids.items() if now - ts > LOCK_TTL]:
        del locked_cam1_ids[k]
    for k in [k for k, v in locked_cam2_map.items() if now - v['ts'] > LOCK_TTL]:
        del locked_cam2_map[k]


def run_cam2():
    log.info("Init YOLO", extra={"event": "yolo_init", "weights": YOLO_MODEL})
    try:
        model = YOLO(YOLO_MODEL)
    except Exception:
        log.error("Failed to load YOLO", exc_info=True, extra={"event": "yolo_error", "weights": YOLO_MODEL})
        return

    # Feature Extractor with a dynamic model path
    extractor = utils.FeatureExtractor(model_path=OSNET_MODEL, logger=log)

    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO, TOPIC_GALLERY])
    producer = utils.get_kafka_producer(BROKER)

    res_writer = utils.MOTResultWriter(
        output_path=config.res_path("cam2"),
        target_width=None,  # producer does not resize
        original_width=1920,
        original_height=1080,
        logger=log,
    )

    tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60, frame_rate=30)
    box_an = sv.BoxAnnotator(thickness=2)

    gallery_db = {}
    locked_cam1_ids = {}  # {cam1_id: lock_ts} -- timestamped so stale locks can be evicted
    locked_cam2_map = {}  # {cam2_id: {'cam1_id': int, 'ts': lock_ts}}

    frame_count = 0
    match_count = 0
    last_hb = time.time()
    hb_frame_mark = 0

    log.info("Cam2 ready (frame-by-frame matcher)", extra={"event": "ready", "broker": BROKER})

    try:
        while True:
            # Heartbeat first, on wall-clock time: fires even when no frames arrive,
            # so a frame-starved consumer still proves it's alive (fps reads 0) instead
            # of looking dead in the logs.
            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                fps = (frame_count - hb_frame_mark) / (now - last_hb)
                log.info("Cam2 stats", extra={"event": "heartbeat", "fps": round(fps, 1),
                                              "frames": frame_count, "matches": match_count,
                                              "gallery_size": len(gallery_db),
                                              "locks": len(locked_cam2_map)})
                last_hb = now
                hb_frame_mark = frame_count
                # Periodic, frame-rate-independent sweep: keeps gallery_db and the
                # lock tables bounded on long / looping runs (also bounds the O(N)
                # gallery scan below to the active window).
                evict_stale_state(gallery_db, locked_cam1_ids, locked_cam2_map, now)

            msg = consumer.poll(0.02)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka message error", extra={"event": "kafka_error", "error": str(msg.error())})
                continue

            if msg.topic() == TOPIC_GALLERY:
                try:
                    data = json.loads(msg.value().decode())
                    # Insert/refresh; stale entries are pruned by evict_stale_state()
                    gallery_db[data['track_id']] = {
                        'feat': np.array(data['feature'], dtype=np.float32),
                        'ts': data['timestamp'],
                        'img_b64': data.get('image_b64')
                    }
                except (json.JSONDecodeError, KeyError):
                    log.debug("Skipping malformed gallery message", extra={"event": "gallery_parse_error"})
                continue

            if msg.topic() == TOPIC_VIDEO and msg.key().decode() == "cam2":
                nparr = np.frombuffer(msg.value(), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1
                ts_now = time.time()
                kafka_frame_idx = frame_count
                try:
                    if msg.headers():
                        meta = json.loads(msg.headers()[0][1].decode())
                        ts_now = meta.get('timestamp', ts_now)
                        # Producer sends frame_idx from 0, GT starts at 1
                        kafka_frame_idx = meta.get('frame_idx', frame_count) + 1
                except (json.JSONDecodeError, IndexError, AttributeError):
                    log.debug("Could not read meta header", extra={"event": "meta_fallback"})

                if frame_count % 3 == 0:
                    # Isolate inference: a corrupt frame or CUDA OOM skips this frame, not the run
                    try:
                        results = model(frame, verbose=False, conf=0.5)[0]
                    except Exception:
                        log.error("YOLO inference failed, skipping frame", exc_info=True,
                                  extra={"event": "inference_error", "frame": frame_count})
                        continue
                    detections = sv.Detections.from_ultralytics(results)
                    detections = utils.merge_truck_boxes(detections, frame.shape)
                    detections = tracker.update_with_detections(detections)

                    for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                        res_writer.write(kafka_frame_idx, tid, xyxy, conf)

                    # Matching logic
                    query_crops, query_indices = [], []
                    for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                        tid = int(tid)
                        if tid in locked_cam2_map:
                            continue  # skip if already matched

                        x1, y1, x2, y2 = map(int, xyxy)
                        if (x2 - x1) * (y2 - y1) < 800:
                            continue

                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        query_crops.append(crop)
                        query_indices.append(i)

                    if query_crops and gallery_db:
                        try:
                            q_feats = extractor.extract_batch(query_crops)
                        except Exception:
                            log.error("Feature extraction failed, skipping frame", exc_info=True,
                                      extra={"event": "extract_error", "frame": frame_count})
                            continue

                        # Filter the gallery
                        valid_items, valid_ids = [], []
                        for gid, gdata in gallery_db.items():
                            if gid in locked_cam1_ids:
                                continue
                            gap = ts_now - gdata['ts']
                            if MIN_TRAVEL_TIME < gap < MAX_TRAVEL_TIME:
                                valid_items.append(gdata['feat'])
                                valid_ids.append(gid)

                        if valid_items:
                            sim = np.dot(q_feats, np.vstack(valid_items).T)
                            for q_idx, row in enumerate(sim):
                                best_g_idx = np.argmax(row)
                                score = row[best_g_idx]
                                matched_c1_id = valid_ids[best_g_idx]

                                if score > SIMILARITY_THRESHOLD and matched_c1_id not in locked_cam1_ids:
                                    c2_tid = int(detections.tracker_id[query_indices[q_idx]])

                                    # Lock (timestamped so it can be evicted later)
                                    locked_cam1_ids[matched_c1_id] = ts_now
                                    locked_cam2_map[c2_tid] = {'cam1_id': matched_c1_id, 'ts': ts_now}

                                    # Send event (cam_source="cam2")
                                    evt = {
                                        "cam_source": "cam2",  # mark the source
                                        "cam1_id": matched_c1_id,
                                        "match_id": c2_tid,   # ID at cam2
                                        "score": float(score),
                                        "timestamp": ts_now,
                                        "cam1_b64": gallery_db[matched_c1_id]['img_b64'],
                                        "match_b64": utils.encode_image_base64(query_crops[q_idx])
                                    }
                                    try:
                                        producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                                    except BufferError:
                                        log.debug("Kafka buffer full, dropping match msg",
                                                  extra={"event": "buffer_full"})
                                        producer.poll(0.1)
                                    match_count += 1
                                    log.info("Match cam2 <-> cam1",
                                             extra={"event": "match", "cam2_id": c2_tid,
                                                    "cam1_id": matched_c1_id, "score": round(float(score), 3)})

                    # UI: draw boxes
                    frame = box_an.annotate(frame, detections)
                    for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                        tid = int(tid)
                        x1, y1 = int(xyxy[0]), int(xyxy[1])
                        if tid in locked_cam2_map:
                            cv2.putText(frame, f"ID_C1: {locked_cam2_map[tid]['cam1_id']}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y1), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                gui.imshow("Cam 2", frame)
                if gui.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        log.info("Ctrl-C received, stopping cam2", extra={"event": "shutdown"})
    finally:
        consumer.close()
        producer.flush(timeout=5.0)  # deliver any queued match events before exit
        res_writer.close()
        gui.destroyAllWindows()
        log.info("Cam2 exited", extra={"event": "exit", "frames": frame_count, "matches": match_count})


if __name__ == "__main__":
    run_cam2()
