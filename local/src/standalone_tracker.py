import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

# --- CONFIG ---
# Anchor paths to the project root (parent of local/) so it runs from any directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_SOURCES = {
    "cam1": os.path.join(_ROOT, "data", "cam1_half.mp4"),
    "cam2": os.path.join(_ROOT, "data", "cam2_half.mp4"),
    "cam3": os.path.join(_ROOT, "data", "cam3_half.mp4"),
}

MODEL_PATHS = {
    "cam1": os.path.join(_ROOT, "weights", "cam1.pt"),
    "cam2": os.path.join(_ROOT, "weights", "cam2.pt"),
    "cam3": os.path.join(_ROOT, "weights", "cam1.pt"),
}

OUTPUT_DIR = os.path.join(_ROOT, "results")  # output folder

# Tracker config (kept the same as the consumers to stay realistic)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TRACK_THRESH = 0.2
MATCH_THRESH = 0.8
BUFFER_SIZE = 60

class MOTResultWriter:
    def __init__(self, output_path):
        # Create the parent directory if it does not exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, 'w')
        print(f"Writing results to: {output_path}")

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
    print(f"\n>>> Processing: {cam_name} | Model: {model_path}")

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
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

    # tqdm for a progress bar
    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1  # frames are 1-based

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
    print(f"Done {cam_name}.")

def main():
    # Step 1: run tracking
    for cam, v_path in VIDEO_SOURCES.items():
        m_path = MODEL_PATHS.get(cam, "yolov8n.pt")
        process_camera(cam, v_path, m_path)

    # Step 2: run evaluation automatically
    print("\n" + "="*40)
    print("SWITCHING TO EVALUATION...")
    print("="*40)

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_metrics.py")
    if os.path.exists(eval_script):
        os.system(f"{sys.executable} {eval_script}")
    else:
        print("eval_metrics.py not found; cannot run evaluation.")

if __name__ == "__main__":
    main()
