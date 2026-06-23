import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils

import config
from log_utils import setup_logging

log = setup_logging("cam3")

# --- CAM 3 CONFIG ---
BROKER = config.BROKER
TOPIC_VIDEO = config.TOPIC_VIDEO
TOPIC_GALLERY = config.TOPIC_GALLERY
TOPIC_MATCH = config.TOPIC_MATCH
GROUP_ID = config.GROUP_CAM3

YOLO_MODEL = config.YOLO_CAM3
OSNET_MODEL = config.OSNET_WEIGHTS

SIMILARITY_THRESHOLD = 0.5
MIN_TRAVEL_TIME = 20
MAX_TRAVEL_TIME = 40

# Number of missing frames before a track is considered exited.
# e.g. 20 consecutive frames without seeing the ID -> assume the vehicle has passed.
EXIT_FRAME_THRESHOLD = 20
HEARTBEAT_SEC = 5.0


def run_cam3():
    log.info("Init YOLO", extra={"event": "yolo_init", "weights": YOLO_MODEL})
    try:
        model = YOLO(YOLO_MODEL)
    except Exception:
        log.error("Failed to load YOLO", exc_info=True, extra={"event": "yolo_error", "weights": YOLO_MODEL})
        return

    try:
        extractor = utils.FeatureExtractor(model_path=OSNET_MODEL, logger=log)
    except Exception:
        log.error("Failed to init FeatureExtractor", exc_info=True, extra={"event": "extractor_error"})
        return

    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO, TOPIC_GALLERY])
    producer = utils.get_kafka_producer(BROKER)

    res_writer = utils.MOTResultWriter(
        output_path="results/res_cam3.txt",
        target_width=None,
        original_width=1440,  # different native resolution
        original_height=1080,
        logger=log,
    )

    tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60, frame_rate=30)
    box_an = sv.BoxAnnotator(thickness=2)

    gallery_db = {}
    locked_cam1_ids = set()

    # Remember vehicles already handled (matched or new) so flicker doesn't reprocess them
    processed_cam3_ids = set()

    # Buffer holding the best shot per track
    # Format: { track_id: {'crop': img, 'area': int, 'ts': float, 'missing_count': int} }
    best_track_buffer = {}

    frame_count = 0
    match_count = 0
    new_count = 0
    last_hb = time.time()
    hb_frame_mark = 0
    log.info("Cam3 ready (best-shot buffering)", extra={"event": "ready", "broker": BROKER})

    try:
        while True:
            msg = consumer.poll(0.02)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka message error", extra={"event": "kafka_error", "error": str(msg.error())})
                continue

            # --- RECEIVE CAM 1 DATA ---
            if msg.topic() == TOPIC_GALLERY:
                try:
                    data = json.loads(msg.value().decode())
                    gallery_db[data['track_id']] = {
                        'feat': np.array(data['feature'], dtype=np.float32),
                        'ts': data['timestamp'],
                        'img_b64': data.get('image_b64')
                    }
                    if len(gallery_db) % 50 == 0:
                        now = time.time()
                        expired = [k for k, v in gallery_db.items() if now - v['ts'] > MAX_TRAVEL_TIME]
                        for k in expired:
                            del gallery_db[k]
                except (json.JSONDecodeError, KeyError):
                    log.debug("Skipping malformed gallery message", extra={"event": "gallery_parse_error"})
                continue

            # --- PROCESS CAM 3 ---
            if msg.topic() == TOPIC_VIDEO and msg.key().decode() == "cam3":
                nparr = np.frombuffer(msg.value(), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1
                ts_now = time.time()
                kafka_frame_idx = frame_count
                try:
                    meta = json.loads(msg.headers()[0][1].decode())
                    kafka_frame_idx = meta.get('frame_idx', frame_count) + 1
                    ts_now = meta.get('timestamp', ts_now)
                except (json.JSONDecodeError, IndexError, AttributeError):
                    log.debug("Could not read meta header", extra={"event": "meta_fallback"})

                # 1. Detect & track
                if frame_count % 3 == 0:
                    results = model(frame, verbose=False, conf=0.5)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    detections = utils.merge_truck_boxes(detections, frame.shape)
                    detections = tracker.update_with_detections(detections)

                    confs = detections.confidence if detections.confidence is not None else [1.0] * len(detections)
                    for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, confs):
                        res_writer.write(kafka_frame_idx, tid, xyxy, conf)

                    # IDs present in this frame
                    current_frame_tids = set()

                    # 2. UPDATE BUFFER (find the largest shot)
                    for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                        tid = int(tid)
                        current_frame_tids.add(tid)

                        # If this vehicle is already handled (matched/new) -> skip
                        if tid in processed_cam3_ids:
                            continue

                        x1, y1, x2, y2 = map(int, xyxy)
                        if (x2 - x1) * (y2 - y1) < 800:
                            continue

                        area = (x2 - x1) * (y2 - y1)
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                        # Best-shot update logic
                        if tid not in best_track_buffer:
                            # Newly appeared
                            best_track_buffer[tid] = {
                                'crop': crop,
                                'area': area,
                                'ts': ts_now,
                                'missing_count': 0
                            }
                        else:
                            # Already tracked, reset missing count since we see it now
                            best_track_buffer[tid]['missing_count'] = 0

                            # If the new shot is larger than the old one -> update
                            if area > best_track_buffer[tid]['area']:
                                best_track_buffer[tid]['crop'] = crop
                                best_track_buffer[tid]['area'] = area
                                best_track_buffer[tid]['ts'] = ts_now  # timestamp of the best shot

                    # 3. CHECK FOR EXITED TRACKS (vehicles that disappeared)
                    finished_tracks = []  # vehicles ready to match
                    buffer_ids_to_remove = []

                    for tid, info in best_track_buffer.items():
                        # If a buffered vehicle is NOT in the current frame
                        if tid not in current_frame_tids:
                            info['missing_count'] += 1

                            # Missing too long -> consider it gone
                            if info['missing_count'] > EXIT_FRAME_THRESHOLD:
                                finished_tracks.append((tid, info))
                                buffer_ids_to_remove.append(tid)

                    # 4. MATCH THE FINISHED TRACKS
                    if finished_tracks:
                        # Batch extract features once for speed
                        crops_to_process = [info['crop'] for _, info in finished_tracks]
                        q_feats = extractor.extract_batch(crops_to_process)

                        # Build the valid gallery list
                        valid_items, valid_ids = [], []
                        if gallery_db:
                            for gid, gdata in gallery_db.items():
                                if gid in locked_cam1_ids:
                                    continue
                                # Compare against the best-shot timestamp (info['ts']);
                                # using ts_now here is also acceptable
                                gap = ts_now - gdata['ts']
                                if MIN_TRAVEL_TIME < gap < MAX_TRAVEL_TIME:
                                    valid_items.append(gdata['feat'])
                                    valid_ids.append(gid)

                        # Match each finished vehicle
                        for idx, (c3_tid, info) in enumerate(finished_tracks):

                            processed_cam3_ids.add(c3_tid)  # mark as handled
                            img_b64 = utils.encode_image_base64(info['crop'])
                            match_found = False

                            if valid_items:
                                # Compute similarity
                                query_feat = q_feats[idx].reshape(1, -1)
                                gallery_feats = np.vstack(valid_items)

                                sim = np.dot(query_feat, gallery_feats.T)[0]
                                best_idx = np.argmax(sim)
                                score = sim[best_idx]
                                matched_c1_id = valid_ids[best_idx]

                                if score > SIMILARITY_THRESHOLD and matched_c1_id not in locked_cam1_ids:
                                    # === MATCH FOUND ===
                                    locked_cam1_ids.add(matched_c1_id)

                                    evt = {
                                        "cam_source": "cam3",
                                        "cam1_id": matched_c1_id,
                                        "match_id": c3_tid,
                                        "score": float(score),
                                        "timestamp": info['ts'],  # use the best-shot timestamp
                                        "match_b64": img_b64,
                                        "is_new": False
                                    }
                                    producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                                    match_count += 1
                                    log.info("Match cam3 <-> cam1",
                                             extra={"event": "match", "cam3_id": c3_tid,
                                                    "cam1_id": matched_c1_id, "score": round(float(score), 3)})
                                    match_found = True

                            if not match_found:
                                # === NEW VEHICLE ===
                                # No match -> new vehicle
                                evt = {
                                    "cam_source": "cam3",
                                    "match_id": c3_tid,
                                    "timestamp": info['ts'],
                                    "match_b64": img_b64,
                                    "is_new": True
                                }
                                producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                                new_count += 1
                                log.info("New vehicle at cam3 (no match)",
                                         extra={"event": "new_vehicle", "cam3_id": c3_tid})

                    # 5. Clean up the buffer
                    for tid in buffer_ids_to_remove:
                        del best_track_buffer[tid]

                    # UI: draw boxes (still drawn while the vehicle is moving)
                    frame = box_an.annotate(frame, detections)
                    for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                        tid = int(tid)
                        x1, y1 = int(xyxy[0]), int(xyxy[1])

                        if tid in best_track_buffer:
                            # Show current area for debugging
                            area = best_track_buffer[tid]['area']
                            cv2.putText(frame, f"ID:{tid} Area:{area}", (x1, y1 - 10), 0, 0.5, (255, 255, 0), 1)
                        elif tid in processed_cam3_ids:
                            cv2.putText(frame, f"ID:{tid} (Done)", (x1, y1 - 10), 0, 0.5, (100, 100, 100), 1)

                # Heartbeat
                now = time.time()
                if now - last_hb >= HEARTBEAT_SEC:
                    fps = (frame_count - hb_frame_mark) / (now - last_hb)
                    log.info("Cam3 stats", extra={"event": "heartbeat", "fps": round(fps, 1),
                                                  "frames": frame_count, "matches": match_count,
                                                  "new": new_count, "tracking": len(best_track_buffer)})
                    last_hb = now
                    hb_frame_mark = frame_count

                cv2.imshow("Cam 3", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        log.info("Ctrl-C received, stopping cam3", extra={"event": "shutdown"})
    finally:
        consumer.close()
        res_writer.close()
        cv2.destroyAllWindows()
        log.info("Cam3 exited", extra={"event": "exit", "frames": frame_count,
                                       "matches": match_count, "new": new_count})


if __name__ == "__main__":
    run_cam3()
