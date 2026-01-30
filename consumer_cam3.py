import cv2
import numpy as np
import json
import time
import datetime
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils

# --- CONFIG CAM 3 ---
BROKER = "localhost:9092"
TOPIC_VIDEO = "video_reid_stream"
TOPIC_GALLERY = "reid_gallery_stream"
TOPIC_MATCH = "reid_matches"
GROUP_ID = "cam3_service_best_shot_v1" # Đổi group ID mới

YOLO_MODEL = "weights/cam1.pt"
OSNET_MODEL = "weights/osnet_cam123.pth"

SIMILARITY_THRESHOLD = 0.5
MIN_TRAVEL_TIME = 20  
MAX_TRAVEL_TIME = 40 

# [NEW] Số frame vắng mặt để xác nhận track đã kết thúc
# Ví dụ: 10 frame liên tiếp (~0.3s) không thấy ID đó nữa thì coi như xe đã đi qua
EXIT_FRAME_THRESHOLD = 20 

def run_cam3():
    print(f"[Cam3] Init...")
    try:
        model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"[ERROR] YOLO: {e}")
        return

    try:
        extractor = utils.FeatureExtractor(model_path=OSNET_MODEL)
    except Exception as e: return

    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO, TOPIC_GALLERY])
    producer = utils.get_kafka_producer(BROKER)

    tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60, frame_rate=30)
    box_an = sv.BoxAnnotator(thickness=2)

    gallery_db = {}
    locked_cam1_ids = set()
    
    # Set để nhớ những xe đã xử lý xong (Match hoặc New) để không xử lý lại nếu nó flicker
    processed_cam3_ids = set()

    # [NEW] Buffer lưu trữ ảnh tốt nhất của từng track
    # Format: { track_id: {'crop': img, 'area': int, 'ts': float, 'missing_count': int} }
    best_track_buffer = {}

    frame_count = 0
    print("[Cam3] Ready")

    while True:
        msg = consumer.poll(0.02)
        if msg is None: continue
        if msg.error(): continue

        # --- NHẬN DATA CAM 1 ---
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
                    expired = [k for k,v in gallery_db.items() if now - v['ts'] > MAX_TRAVEL_TIME]
                    for k in expired: del gallery_db[k]
            except: pass
            continue

        # --- XỬ LÝ CAM 3 ---
        if msg.topic() == TOPIC_VIDEO and msg.key().decode() == "cam3":
            nparr = np.frombuffer(msg.value(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue
            
            frame_count += 1
            ts_now = time.time()
            try:
                meta = json.loads(msg.headers()[0][1].decode())
                ts_now = meta.get('timestamp', ts_now)
            except: pass

            # 1. Detect & Track
            if frame_count % 3 == 0:
                results = model(frame, verbose=False, conf=0.5)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = utils.merge_truck_boxes(detections, frame.shape)
                detections = tracker.update_with_detections(detections)

                # Tập hợp các ID đang xuất hiện trong khung hình này
                current_frame_tids = set()

                # 2. CẬP NHẬT BUFFER (Tìm ảnh to nhất)
                for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                    tid = int(tid)
                    current_frame_tids.add(tid)

                    # Nếu xe này đã xử lý xong (đã match/new trước đó) -> Bỏ qua
                    if tid in processed_cam3_ids: continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    if (x2-x1)*(y2-y1) < 800: continue
                    
                    area = (x2-x1) * (y2-y1)
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]

                    # Logic cập nhật Best Shot
                    if tid not in best_track_buffer:
                        # Mới xuất hiện
                        best_track_buffer[tid] = {
                            'crop': crop,
                            'area': area,
                            'ts': ts_now,
                            'missing_count': 0
                        }
                    else:
                        # Đã có, reset missing count vì đang nhìn thấy
                        best_track_buffer[tid]['missing_count'] = 0
                        
                        # Nếu ảnh mới to hơn ảnh cũ -> Cập nhật
                        if area > best_track_buffer[tid]['area']:
                            best_track_buffer[tid]['crop'] = crop
                            best_track_buffer[tid]['area'] = area
                            best_track_buffer[tid]['ts'] = ts_now # Update timestamp lúc ảnh đẹp nhất

                # 3. KIỂM TRA TRACK KẾT THÚC (Xử lý xe biến mất)
                # Duyệt qua các xe đang theo dõi trong buffer
                finished_tracks = [] # List các xe cần đem đi match
                buffer_ids_to_remove = []

                for tid, info in best_track_buffer.items():
                    # Nếu xe trong buffer KHÔNG có trong khung hình hiện tại
                    if tid not in current_frame_tids:
                        info['missing_count'] += 1
                        
                        # Nếu vắng mặt quá lâu -> Coi như đã đi xong
                        if info['missing_count'] > EXIT_FRAME_THRESHOLD:
                            finished_tracks.append((tid, info))
                            buffer_ids_to_remove.append(tid)

                # 4. THỰC HIỆN MATCHING CHO CÁC XE ĐÃ KẾT THÚC
                if finished_tracks:
                    # Gom batch để extract feature 1 lần cho nhanh
                    crops_to_process = [info['crop'] for _, info in finished_tracks]
                    q_feats = extractor.extract_batch(crops_to_process)
                    
                    # Lấy danh sách Gallery hợp lệ
                    valid_items, valid_ids = [], []
                    if gallery_db:
                        for gid, gdata in gallery_db.items():
                            if gid in locked_cam1_ids: continue
                            # Lưu ý: So sánh với timestamp của tấm ảnh đẹp nhất (info['ts'])
                            # Nhưng ở đây lấy ts_now cũng tạm ổn
                            gap = ts_now - gdata['ts'] 
                            if MIN_TRAVEL_TIME < gap < MAX_TRAVEL_TIME:
                                valid_items.append(gdata['feat'])
                                valid_ids.append(gid)

                    # Duyệt qua từng xe đã kết thúc để Match
                    for idx, (c3_tid, info) in enumerate(finished_tracks):
                        
                        processed_cam3_ids.add(c3_tid) # Đánh dấu đã xử lý
                        img_b64 = utils.encode_image_base64(info['crop'])
                        match_found = False

                        if valid_items:
                            # Tính Sim
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
                                    "timestamp": info['ts'], # Dùng TS lúc chụp ảnh đẹp nhất
                                    "match_b64": img_b64,
                                    "is_new": False
                                }
                                producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                                print(f"✅ [CAM3] MATCH vehicle ID {c3_tid} with ID {matched_c1_id} (cam1)")
                                match_found = True

                        if not match_found:
                            # === NEW VEHICLE ===
                            # Không match ai cả -> Xe mới
                            evt = {
                                "cam_source": "cam3",
                                "match_id": c3_tid,
                                "timestamp": info['ts'],
                                "match_b64": img_b64,
                                "is_new": True 
                            }
                            producer.produce(TOPIC_MATCH, json.dumps(evt).encode())
                            print(f"🆕 [CAM3] NEW Vehicle ID {c3_tid}")

                # 5. Dọn dẹp Buffer
                for tid in buffer_ids_to_remove:
                    del best_track_buffer[tid]

                # UI Vẽ Box (Vẫn vẽ bình thường khi xe đang chạy)
                frame = box_an.annotate(frame, detections)
                for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                    tid = int(tid)
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    
                    if tid in best_track_buffer:
                        # Hiển thị Area hiện tại để debug
                        area = best_track_buffer[tid]['area']
                        cv2.putText(frame, f"ID:{tid} Area:{area}", (x1, y1-10), 0, 0.5, (255,255,0), 1)
                    elif tid in processed_cam3_ids:
                        cv2.putText(frame, f"ID:{tid} (Done)", (x1, y1-10), 0, 0.5, (100,100,100), 1)

            cv2.imshow("Cam 3", frame)
            if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    run_cam3()