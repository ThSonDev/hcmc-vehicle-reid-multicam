import cv2
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
SKIP_FRAMES = 2                 # run YOLO every (SKIP_FRAMES + 1)th frame
YOLO_CONF = 0.50               # YOLO confidence threshold
MIN_AREA_THRESHOLD = 800       # min crop area (px) to embed
SEND_INTERVAL = 0.5            # min seconds between gallery publishes per track
TRACK_THRESH = 0.2            # ByteTrack activation threshold
MATCH_THRESH = 0.8           # ByteTrack matching threshold
LOST_TRACK_BUFFER = 30      # ByteTrack frames a track survives while unseen
DATA_FPS = 30               # source frame rate (for ByteTrack)
POLL_TIMEOUT = 0.01         # Kafka poll timeout (s)
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
        lost_track_buffer=LOST_TRACK_BUFFER,
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
    hb = utils.Heartbeat(log, "Cam1", HEARTBEAT_SEC)

    log.info("Cam1 ready (gallery source)", extra={"event": "ready", "broker": BROKER})

    try:
        while True:
            hb.tick(frame_count, gallery_sent=sent_count, tracks=len(cam1_last_sent))

            msg = consumer.poll(POLL_TIMEOUT)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka message error", extra={"event": "kafka_error", "error": str(msg.error())})
                continue

            # Only process cam1 messages
            if msg.key().decode() != "cam1":
                continue

            frame = utils.decode_frame(msg)
            if frame is None:
                continue

            frame_count += 1
            ts_now, kafka_frame_idx = utils.parse_frame_meta(msg, frame_count, log)

            time_str = datetime.datetime.fromtimestamp(ts_now).strftime('%H:%M:%S.%f')[:-4]
            detections_display = None

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
                detections_display = detections

                res_writer.write_detections(kafka_frame_idx, detections)

                crops, valid_indices = [], []
                for i, xyxy in enumerate(detections.xyxy):
                    crop, area = utils.clamp_crop(frame, xyxy)
                    if area > MIN_AREA_THRESHOLD:
                        crops.append(crop)
                        valid_indices.append(i)

                if crops:
                    try:
                        embeddings = extractor.extract_batch(crops)
                    except Exception:
                        log.error("Feature extraction failed, skipping frame", exc_info=True,
                                  extra={"event": "extract_error", "frame": frame_count})
                        continue

                    for k, (idx, feat_vec) in enumerate(zip(valid_indices, embeddings)):
                        tid = int(detections.tracker_id[idx])
                        if ts_now - cam1_last_sent.get(tid, 0) <= SEND_INTERVAL:
                            continue
                        payload = {
                            "track_id": tid,
                            "timestamp": ts_now,
                            "feature": feat_vec.tolist(),
                            "cam_id": "cam1",
                            "image_b64": utils.encode_image_base64(crops[k]),
                        }
                        if not utils.produce_event(producer, TOPIC_GALLERY, payload, log, key=str(tid).encode()):
                            continue  # queue full: retry this track next interval
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
        producer.flush(timeout=5.0)  # deliver any queued gallery embeddings before exit
        res_writer.close()
        gui.destroyAllWindows()
        log.info("Cam1 exited", extra={"event": "exit", "frames": frame_count, "gallery_sent": sent_count})


if __name__ == "__main__":
    run_cam1()
