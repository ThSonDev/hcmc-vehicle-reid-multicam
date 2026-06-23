"""Shared config for the local pipeline (Kafka, topics, group IDs, model paths).

Centralizes the infra constants that were duplicated across all 5 scripts. Per-camera
tuning params stay in each consumer for easy tweaking. Overridable via env vars
(handy for Docker / Airflow / tests).
"""
import os

# --- KAFKA ---
BROKER = os.environ.get("REID_BROKER", "localhost:9092")

TOPIC_VIDEO = "video_reid_stream"
TOPIC_GALLERY = "reid_gallery_stream"
TOPIC_MATCH = "reid_matches"

# --- CONSUMER GROUP IDS ---
GROUP_CAM1 = "cam1_service_v2"
GROUP_CAM2 = "cam2_service_v2"
GROUP_CAM3 = "cam3_service_best_shot_v1"
GROUP_VISUALIZER = "visualizer_bottom_v4"

# --- MODEL PATHS ---
YOLO_CAM1 = "weights/cam1.pt"
YOLO_CAM2 = "weights/cam2.pt"
YOLO_CAM3 = "weights/cam1.pt"          # cam3 currently reuses cam1's weights
OSNET_WEIGHTS = "weights/osnet_cam123.pth"

# --- VIDEO SOURCES (producer) ---
VIDEO_SOURCES = {
    "cam1": "data/cam1_640.mp4",
    "cam2": "data/cam2_640.mp4",
    "cam3": "data/cam3_640.mp4",
}

# --- DIRECTORIES ---
LOG_DIR = os.environ.get("REID_LOG_DIR", "logs")
RESULTS_DIR = "results"
