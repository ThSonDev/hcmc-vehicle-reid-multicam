import cv2
import numpy as np
import time
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils

import config
import gui_utils as gui
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

# --- TUNING ---
SKIP_FRAMES = 2               # run YOLO every (SKIP_FRAMES + 1)th frame
YOLO_CONF = 0.5             # YOLO confidence threshold
MIN_AREA_THRESHOLD = 800   # min crop area (px) to buffer/match
TRACK_THRESH = 0.2         # ByteTrack activation threshold
LOST_TRACK_BUFFER = 60     # ByteTrack frames a track survives while unseen
DATA_FPS = 30              # source frame rate (for ByteTrack)
POLL_TIMEOUT = 0.02        # Kafka poll timeout (s)
GALLERY_EVICT_EVERY = 50   # prune the gallery every N inserts

SIMILARITY_THRESHOLD = 0.5   # cosine-similarity bar (lower than cam2: harder cross-road match)
MIN_TRAVEL_TIME = 20        # cam1->cam3 travel-time gate (s)
MAX_TRAVEL_TIME = 40

# Missing *processed* frames before a track is treated as exited. With SKIP_FRAMES=2 the
# pipeline processes 1 of every 3 captured frames, so 20 processed frames ~= 60 captured
# frames (~2s at 30 fps) -- roughly in line with ByteTrack's lost_track_buffer.
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
        output_path=config.res_path("cam3"),
        target_width=None,
        original_width=1440,  # different native resolution
        original_height=1080,
        logger=log,
    )

    tracker = sv.ByteTrack(track_activation_threshold=TRACK_THRESH,
                           lost_track_buffer=LOST_TRACK_BUFFER, frame_rate=DATA_FPS)
    box_an = sv.BoxAnnotator(thickness=2)

    gallery_db = {}
    locked_cam1_ids = set()

    # Remember vehicles already handled (matched or new) so flicker doesn't reprocess them
    processed_cam3_ids = set()

    # Best shot per track: { track_id: {'crop', 'area', 'ts', 'missing_count'} }
    best_track_buffer = {}

    frame_count = 0
    match_count = 0
    new_count = 0
    hb = utils.Heartbeat(log, "Cam3", HEARTBEAT_SEC)
    log.info("Cam3 ready (best-shot buffering)", extra={"event": "ready", "broker": BROKER})

    try:
        while True:
            hb.tick(frame_count, matches=match_count, new=new_count, tracking=len(best_track_buffer))

            msg = consumer.poll(POLL_TIMEOUT)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka message error", extra={"event": "kafka_error", "error": str(msg.error())})
                continue

            if msg.topic() == TOPIC_GALLERY:
                if utils.ingest_gallery_message(gallery_db, msg, log) and len(gallery_db) % GALLERY_EVICT_EVERY == 0:
                    now = time.time()
                    for k in [k for k, v in gallery_db.items() if now - v['ts'] > MAX_TRAVEL_TIME]:
                        del gallery_db[k]
                continue

            if msg.topic() == TOPIC_VIDEO and msg.key().decode() == "cam3":
                frame = utils.decode_frame(msg)
                if frame is None:
                    continue

                frame_count += 1
                ts_now, kafka_frame_idx = utils.parse_frame_meta(msg, frame_count, log)

                if frame_count % (SKIP_FRAMES + 1) == 0:
                    # Isolate inference: a corrupt frame or CUDA OOM skips this frame, not the run
                    try:
                        results = model(frame, verbose=False, conf=YOLO_CONF)[0]
                    except Exception:
                        log.error("YOLO inference failed, skipping frame", exc_info=True,
                                  extra={"event": "inference_error", "frame": frame_count})
                        continue
                    detections = sv.Detections.from_ultralytics(results)
                    detections = utils.merge_truck_boxes(detections, frame.shape)
                    detections = tracker.update_with_detections(detections)

                    res_writer.write_detections(kafka_frame_idx, detections)

                    # Update the best-shot buffer (keep the largest crop per track)
                    current_frame_tids = set()
                    for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                        tid = int(tid)
                        current_frame_tids.add(tid)
                        if tid in processed_cam3_ids:
                            continue

                        crop, area = utils.clamp_crop(frame, xyxy)
                        if area < MIN_AREA_THRESHOLD:
                            continue

                        if tid not in best_track_buffer:
                            best_track_buffer[tid] = {'crop': crop, 'area': area,
                                                      'ts': ts_now, 'missing_count': 0}
                        else:
                            best_track_buffer[tid]['missing_count'] = 0
                            if area > best_track_buffer[tid]['area']:
                                best_track_buffer[tid]['crop'] = crop
                                best_track_buffer[tid]['area'] = area
                                best_track_buffer[tid]['ts'] = ts_now  # timestamp of the best shot

                    # A buffered track unseen for EXIT_FRAME_THRESHOLD processed frames has exited
                    finished_tracks = []
                    buffer_ids_to_remove = []
                    for tid, info in best_track_buffer.items():
                        if tid not in current_frame_tids:
                            info['missing_count'] += 1
                            if info['missing_count'] > EXIT_FRAME_THRESHOLD:
                                finished_tracks.append((tid, info))
                                buffer_ids_to_remove.append(tid)

                    # Match the exited tracks against the gallery, once
                    if finished_tracks:
                        crops_to_process = [info['crop'] for _, info in finished_tracks]
                        try:
                            q_feats = extractor.extract_batch(crops_to_process)
                        except Exception:
                            # Skip this frame; finished tracks stay buffered and retry next pass
                            log.error("Feature extraction failed, skipping finished tracks", exc_info=True,
                                      extra={"event": "extract_error", "frame": frame_count})
                            continue

                        # Gallery entries within the cam1->cam3 travel-time window
                        valid_items, valid_ids = [], []
                        for gid, gdata in gallery_db.items():
                            if gid in locked_cam1_ids:
                                continue
                            if MIN_TRAVEL_TIME < ts_now - gdata['ts'] < MAX_TRAVEL_TIME:
                                valid_items.append(gdata['feat'])
                                valid_ids.append(gid)

                        for idx, (c3_tid, info) in enumerate(finished_tracks):
                            processed_cam3_ids.add(c3_tid)  # mark as handled
                            img_b64 = utils.encode_image_base64(info['crop'])
                            match_found = False

                            if valid_items:
                                sim = np.dot(q_feats[idx].reshape(1, -1), np.vstack(valid_items).T)[0]
                                best_idx = np.argmax(sim)
                                score = sim[best_idx]
                                matched_c1_id = valid_ids[best_idx]

                                if score > SIMILARITY_THRESHOLD and matched_c1_id not in locked_cam1_ids:
                                    locked_cam1_ids.add(matched_c1_id)
                                    evt = {
                                        "cam_source": "cam3",
                                        "cam1_id": matched_c1_id,
                                        "match_id": c3_tid,
                                        "score": float(score),
                                        "timestamp": info['ts'],  # best-shot timestamp
                                        "cam1_b64": gallery_db[matched_c1_id]['img_b64'],
                                        "match_b64": img_b64,
                                        "is_new": False,
                                    }
                                    utils.produce_event(producer, TOPIC_MATCH, evt, log)
                                    match_count += 1
                                    log.info("Match cam3 <-> cam1",
                                             extra={"event": "match", "cam3_id": c3_tid,
                                                    "cam1_id": matched_c1_id, "score": round(float(score), 3)})
                                    match_found = True

                            if not match_found:
                                # No gallery match within the window -> report a new vehicle
                                evt = {
                                    "cam_source": "cam3",
                                    "match_id": c3_tid,
                                    "timestamp": info['ts'],
                                    "match_b64": img_b64,
                                    "is_new": True,
                                }
                                utils.produce_event(producer, TOPIC_MATCH, evt, log)
                                new_count += 1
                                log.info("New vehicle at cam3 (no match)",
                                         extra={"event": "new_vehicle", "cam3_id": c3_tid})

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

                gui.imshow("Cam 3", frame)
                if gui.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        log.info("Ctrl-C received, stopping cam3", extra={"event": "shutdown"})
    finally:
        consumer.close()
        producer.flush(timeout=5.0)  # deliver any queued match/new-vehicle events before exit
        res_writer.close()
        gui.destroyAllWindows()
        log.info("Cam3 exited", extra={"event": "exit", "frames": frame_count,
                                       "matches": match_count, "new": new_count})


if __name__ == "__main__":
    run_cam3()
