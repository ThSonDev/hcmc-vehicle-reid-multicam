import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

# --- CẤU HÌNH ---
VIDEO_SOURCES = {
    "cam1": "data/cam1_half.mp4",
    "cam2": "data/cam2_half.mp4",
    "cam3": "data/cam3_half.mp4"
}

MODEL_PATHS = {
    "cam1": "weights/cam1.pt",
    "cam2": "weights/cam2.pt",
    "cam3": "weights/cam1.pt" 
}

OUTPUT_DIR = "results"  # Folder chứa kết quả

# Tracker Config (Cần giống Consumer để sát thực tế)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TRACK_THRESH = 0.2
MATCH_THRESH = 0.8
BUFFER_SIZE = 60

class MOTResultWriter:
    def __init__(self, output_path):
        # Tự động tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, 'w')
        print(f"📄 Đang ghi log vào: {output_path}")

    def write(self, frame_idx, track_id, xyxy, conf=1.0):
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        # MOT Format: frame(1-based), id, left, top, w, h, conf, -1, -1, -1
        line = f"{int(frame_idx)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
        self.file.write(line)

    def close(self):
        self.file.close()

def process_camera(cam_name, video_path, model_path):
    print(f"\n>>> 🎥 XỬ LÝ: {cam_name} | Model: {model_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    model = YOLO(model_path)
    
    # Init ByteTrack
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_THRESH,
        lost_track_buffer=BUFFER_SIZE,
        minimum_matching_threshold=MATCH_THRESH,
        frame_rate=30
    )

    out_file = os.path.join(OUTPUT_DIR, f"res_{cam_name}.txt")
    writer = MOTResultWriter(out_file)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0 
    
    # Dùng tqdm để hiện thanh tiến trình
    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1 # Frame bắt đầu từ 1

            # 1. Detect
            results = model(frame, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
            detections = sv.Detections.from_ultralytics(results)

            # 2. Track
            detections = tracker.update_with_detections(detections)

            # 3. Log Result
            confs = detections.confidence if detections.confidence is not None else [1.0] * len(detections)
            for xyxy, tid, conf in zip(detections.xyxy, detections.tracker_id, confs):
                writer.write(frame_idx, tid, xyxy, conf)
            
            pbar.update(1)

    cap.release()
    writer.close()
    print(f"✅ Xong {cam_name}.")

def main():
    # Bước 1: Chạy Tracking
    for cam, v_path in VIDEO_SOURCES.items():
        m_path = MODEL_PATHS.get(cam, "yolov8n.pt")
        process_camera(cam, v_path, m_path)

    # Bước 2: Tự động chạy đánh giá
    print("\n" + "="*40)
    print("🚀 CHUYỂN SANG ĐÁNH GIÁ (EVALUATION)...")
    print("="*40)
    
    if os.path.exists("eval_metrics.py"):
        os.system("python eval_metrics.py")
    else:
        print("❌ Không tìm thấy file eval_metrics.py để chạy đánh giá.")

if __name__ == "__main__":
    main()