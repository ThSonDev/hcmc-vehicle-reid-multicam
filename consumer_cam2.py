import cv2
import numpy as np
import json
import time
import datetime
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils # Import module utils mới

# --- CONFIG ---
BROKER = "localhost:9092"
TOPIC_VIDEO = "video_reid_stream"
TOPIC_GALLERY = "reid_gallery_stream"
TOPIC_MATCH = "reid_matches"
GROUP_ID = "cam2_service_v2"

YOLO_MODEL = "weights/cam2.pt"
# [NEW] Đường dẫn model OSNet riêng cho Cam2
OSNET_MODEL = "weights/osnet_cam123.pth" 

SIMILARITY_THRESHOLD = 0.65
MIN_TRAVEL_TIME = 1.0
MAX_TRAVEL_TIME = 10

def run_cam2():
    print(f"[Cam2] Init...")
    try:
        model = YOLO(YOLO_MODEL)
    except: return

    # Feature Extractor với model path động
    extractor = utils.FeatureExtractor(model_path=OSNET_MODEL)
    
    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO, TOPIC_GALLERY])
    producer = utils.get_kafka_producer(BROKER)

    tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60, frame_rate=30)
    box_an = sv.BoxAnnotator(thickness=2)

    gallery_db = {}
    locked_cam1_ids = set()
    locked_cam2_map = {} # {cam2_id: cam1_id}

    frame_count = 0

    print("[Cam2] Ready")
    
    while True:
        msg = consumer.poll(0.02)
        if msg is None: continue
        if msg.error(): continue

        if msg.topic() == TOPIC_GALLERY:
            try:
                data = json.loads(msg.value().decode())
                # Cam2 không xóa DB, chỉ update
                gallery_db[data['track_id']] = {
                    'feat': np.array(data['feature'], dtype=np.float32),
                    'ts': data['timestamp'],
                    'img_b64': data.get('image_b64')
                }
            except: pass
            continue

        if msg.topic() == TOPIC_VIDEO and msg.key().decode() == "cam2":
            nparr = np.frombuffer(msg.value(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue
            
            frame_count += 1
            ts_now = time.time()
            try:
                meta = json.loads(msg.headers()[0][1].decode())
                ts_now = meta.get('timestamp', ts_now)
            except: pass

            if frame_count % 3 == 0:
                results = model(frame, verbose=False, conf=0.5)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = utils.merge_truck_boxes(detections, frame.shape)
                detections = tracker.update_with_detections(detections)

                # Logic Match
                query_crops, query_indices = [], []
                for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                    tid = int(tid)
                    if tid in locked_cam2_map: continue # Skip nếu đã match

                    x1, y1, x2, y2 = map(int, xyxy)
                    if (x2-x1)*(y2-y1) < 800: continue
                    
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    query_crops.append(crop)
                    query_indices.append(i)

                if query_crops and gallery_db:
                    q_feats = extractor.extract_batch(query_crops)
                    
                    # Filter Gallery
                    valid_items, valid_ids = [], []
                    for gid, gdata in gallery_db.items():
                        if gid in locked_cam1_ids: continue
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
                                
                                # Lock
                                locked_cam1_ids.add(matched_c1_id)
                                locked_cam2_map[c2_tid] = matched_c1_id
                                
                                # Send Event (cam_source="cam2")
                                evt = {
                                    "cam_source": "cam2", # Đánh dấu nguồn
                                    "cam1_id": matched_c1_id,
                                    "match_id": c2_tid,   # ID tại Cam2
                                    "score": float(score),
                                    "timestamp": ts_now,
                                    "cam1_b64": gallery_db[matched_c1_id]['img_b64'],
                                    "match_b64": utils.encode_image_base64(query_crops[q_idx])
                                }
                                producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                                print(f"✅ [CAM2] Match C2#{c2_tid} <-> C1#{matched_c1_id} ({score:.2f})")

                # UI Vẽ Box
                frame = box_an.annotate(frame, detections)
                for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                    tid = int(tid)
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    if tid in locked_cam2_map:
                        cv2.putText(frame, f"ID_C1: {locked_cam2_map[tid]}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.rectangle(frame, (x1,y1), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
            
            cv2.imshow("Cam 2", frame)
            if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    run_cam2()