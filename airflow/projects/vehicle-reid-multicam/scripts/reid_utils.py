import torch
import numpy as np
import cv2
import supervision as sv
import base64
import os
import psycopg2
from confluent_kafka import Consumer, Producer
import torchreid
from torchvision import transforms
from PIL import Image

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config DB (Sử dụng service name 'postgres' trong docker network)
DB_HOST = "postgres"
DB_NAME = "airflow"
DB_USER = "airflow"
DB_PASS = "airflow"

# Config Path
STATIC_DIR = "/opt/airflow/projects/vehicle-reid-multicam/static/captured_vehicles"

# --- DATABASE & STORAGE UTILS ---

def get_db_connection():
    """Tạo kết nối tới Postgres"""
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    return conn

def init_db_tables():
    """Khởi tạo bảng nếu chưa tồn tại"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Bảng 1: Lưu thông tin chi tiết từng Track (Cam1, Cam2, Cam3 đều ghi vào đây)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_tracks (
            global_track_id VARCHAR(50) PRIMARY KEY, -- Format: camX_trackID
            cam_id VARCHAR(10),
            track_id INT,
            timestamp FLOAT,
            img_path TEXT,
            feature_vector FLOAT[], -- Lưu vector ReID
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Bảng 2: Lưu kết quả Match (Streamlit sẽ query bảng này)
    # Logic: Cam 1 luôn là gốc. Cam 2, 3 update vào các cột tương ứng.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_matches (
            cam1_track_id INT PRIMARY KEY,
            cam2_track_id INT DEFAULT NULL,
            cam3_track_id INT DEFAULT NULL,
            cam2_score FLOAT DEFAULT 0,
            cam3_score FLOAT DEFAULT 0,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Tables initialized.")

def save_image_static(img_numpy, cam_id, track_id):
    """Lưu ảnh vào disk, trả về path tương đối"""
    # Tạo folder theo ngày để tránh quá tải folder
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(STATIC_DIR, date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{cam_id}_{track_id}.jpg"
    full_path = os.path.join(save_dir, filename)
    
    # Overwrite nếu có ảnh đẹp hơn
    cv2.imwrite(full_path, img_numpy)
    
    # Trả về path tương đối để lưu DB (dễ migrate)
    return os.path.join("captured_vehicles", date_str, filename)

# --- AI & KAFKA UTILS (Giữ nguyên logic cũ nhưng tối ưu) ---

def encode_image_base64(img_numpy, quality=60):
    _, buffer = cv2.imencode('.jpg', img_numpy, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image_base64(b64_string):
    if not b64_string: return None
    try:
        nparr = np.frombuffer(base64.b64decode(b64_string), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: return None

def merge_truck_boxes(detections, frame_shape):
    # (Giữ nguyên code merge box của bạn)
    if len(detections) < 2: return detections
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
    
    BIG_VEHICLE_CLASSES = [5, 7] 
    merged_indices = set()
    new_xyxy, new_conf, new_cls, new_ids = [], [], [], []
    sorted_idx = np.argsort(xyxy[:, 0])
    
    for i in range(len(sorted_idx)):
        idx_curr = sorted_idx[i]
        if idx_curr in merged_indices: continue
        current_box = xyxy[idx_curr].copy()
        current_cls = class_id[idx_curr]
        
        if current_cls not in BIG_VEHICLE_CLASSES:
            new_xyxy.append(current_box)
            new_conf.append(confidence[idx_curr])
            new_cls.append(current_cls)
            new_ids.append(tracker_id[idx_curr])
            continue

        for j in range(i + 1, len(sorted_idx)):
            idx_next = sorted_idx[j]
            if idx_next in merged_indices: continue
            if class_id[idx_next] not in BIG_VEHICLE_CLASSES: continue
            next_box = xyxy[idx_next]
            gap = max(0, next_box[0] - current_box[2])
            y_min = max(current_box[1], next_box[1])
            y_max = min(current_box[3], next_box[3])
            overlap_h = max(0, y_max - y_min)
            min_height = min(current_box[3]-current_box[1], next_box[3]-next_box[1])
            if gap < 50 and overlap_h > 0.6 * min_height:
                current_box[0] = min(current_box[0], next_box[0])
                current_box[1] = min(current_box[1], next_box[1])
                current_box[2] = max(current_box[2], next_box[2])
                current_box[3] = max(current_box[3], next_box[3])
                merged_indices.add(idx_next)

        new_xyxy.append(current_box)
        new_conf.append(confidence[idx_curr])
        new_cls.append(current_cls)
        new_ids.append(tracker_id[idx_curr])
        merged_indices.add(idx_curr)

    if not new_xyxy: return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(new_xyxy),
        confidence=np.array(new_conf),
        class_id=np.array(new_cls),
        tracker_id=np.array(new_ids)
    )

class FeatureExtractor:
    def __init__(self, model_path, model_arch='osnet_x1_0'):
        print(f"[FeatureExtractor] Init: {model_arch} on {DEVICE}")
        self.model = torchreid.models.build_model(
            name=model_arch, num_classes=100, loss='triplet', pretrained=False
        )
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                state_dict = checkpoint.get('state_dict', checkpoint)
                filtered_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
                self.model.load_state_dict(filtered_dict, strict=False)
            except Exception as e:
                print(f"[ERROR] Weights load failed: {e}")
        self.model.to(DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_batch(self, crops):
        if not crops: return np.array([])
        batch_tensors = []
        for img in crops:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            batch_tensors.append(self.transform(img_pil))
        batch_t = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            features = self.model(batch_t)
            norm = features.norm(p=2, dim=1, keepdim=True)
            features = features.div(norm.expand_as(features))
        return features.cpu().numpy()