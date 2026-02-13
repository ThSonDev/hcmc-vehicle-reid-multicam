import cv2
import numpy as np
import json
import time
import datetime
from ultralytics import YOLO
import supervision as sv
import reid_utils as utils # Sử dụng utils mới

# --- CẤU HÌNH ---
BROKER = "localhost:9092"
TOPIC_VIDEO = "video_reid_stream"
TOPIC_GALLERY = "reid_gallery_stream"
GROUP_ID = "cam1_service_v2"

# Model Paths
YOLO_MODEL_PATH = "weights/cam1.pt"
# [NEW] Đường dẫn model OSNet riêng cho Cam 1 (Nếu bạn chưa train riêng thì dùng osnet_x1_0_imagenet.pth gốc)
OSNET_MODEL_PATH = "weights/osnet_cam123.pth" 

# --- TUNING ---
SKIP_FRAMES = 2             
YOLO_CONF = 0.50             
MIN_AREA_THRESHOLD = 800     
SEND_INTERVAL = 0.5          
TRACK_THRESH = 0.2         
MATCH_THRESH = 0.8
DATA_FPS = 30          

def run_cam1():
    print(f"[Cam1] Init YOLO from {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except:
        print(f"[ERROR] Kiểm tra lại đường dẫn model: {YOLO_MODEL_PATH}")
        return

    res_writer = utils.MOTResultWriter(
        output_path="results/res_cam1.txt", 
        target_width=None, # Hoặc 640 nếu Producer config 640
        original_width=1920, 
        original_height=1080
    )

    # Init Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_THRESH,
        lost_track_buffer=30,
        minimum_matching_threshold=MATCH_THRESH,
        frame_rate=DATA_FPS
    )
    
    # [REFACTOR] Init Feature Extractor với path động
    # Nếu file weight không tồn tại, nó sẽ in warning và dùng weight ngẫu nhiên (hoặc imagenet mặc định nếu chỉnh code)
    extractor = utils.FeatureExtractor(model_path=OSNET_MODEL_PATH)
    
    # [REFACTOR] Init Kafka qua Utils
    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_VIDEO])
    producer = utils.get_kafka_producer(BROKER)
    
    box_an = sv.BoxAnnotator(thickness=2)
    
    frame_count = 0
    cam1_last_sent = {} # Cache: {track_id: timestamp}

    print(f"[Cam1] Ready")
    
    while True:
        msg = consumer.poll(0.01)
        if msg is None: continue
        if msg.error(): continue
        
        # Chỉ xử lý message của cam1
        if msg.key().decode() != "cam1": continue

        nparr = np.frombuffer(msg.value(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: continue

        frame_count += 1
        
        # --- TIMESTAMP CHÍNH XÁC ---
        ts_now = time.time()
        try:
            headers = msg.headers()
            if headers:
                meta = json.loads(headers[0][1].decode())
                ts_now = meta.get('timestamp', ts_now)
        except: pass
        
        dt_object = datetime.datetime.fromtimestamp(ts_now)
        time_str = dt_object.strftime('%H:%M:%S.%f')[:-4]

        detections_display = None

        kafka_frame_idx = frame_count 
        try:
            if msg.headers():
                meta = json.loads(msg.headers()[0][1].decode())
                # Producer gửi frame_idx bắt đầu từ 0, GT thường bắt đầu từ 1
                kafka_frame_idx = meta.get('original_frame_idx', frame_count) + 1
        except: pass

        if frame_count % (SKIP_FRAMES + 1) == 0:
            results = model(frame, verbose=False, conf=YOLO_CONF)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # [REFACTOR] Dùng hàm từ utils
            detections = utils.merge_truck_boxes(detections, frame.shape)
            detections = tracker.update_with_detections(detections)
            detections_display = detections

            for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
                res_writer.write(kafka_frame_idx, tid, xyxy, conf)

            # Chuẩn bị crop
            crops = []
            valid_indices = []
            
            for i, (xyxy, tid) in enumerate(zip(detections.xyxy, detections.tracker_id)):
                x1, y1, x2, y2 = map(int, xyxy)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                area = (x2-x1) * (y2-y1)
                
                if area > MIN_AREA_THRESHOLD:
                    crops.append(frame[y1:y2, x1:x2])
                    valid_indices.append(i)

            if crops:
                # [REFACTOR] Gọi hàm extract từ instance extractor
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
                            # [REFACTOR] Dùng hàm encode từ utils
                            "image_b64": utils.encode_image_base64(crops[valid_indices.index(idx)])
                        }
                        
                        producer.produce(
                            TOPIC_GALLERY,
                            key=str(tid).encode(),
                            value=json.dumps(payload).encode()
                        )
                        cam1_last_sent[tid] = ts_now
                
                producer.poll(0)

        # UI
        if detections_display:
            frame = box_an.annotate(frame, detections_display)
            
        cv2.putText(frame, f"Cam 1: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Cam 1 - Sender", frame)
        if cv2.waitKey(1) == ord('q'): break

    consumer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam1()