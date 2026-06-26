"""Shared config for the local pipeline (Kafka, topics, group IDs, model paths).

Centralizes the infra constants that were duplicated across all 5 scripts. Per-camera
tuning params stay in each consumer for easy tweaking. Overridable via env vars
(handy for Docker / Airflow / tests).
"""
import os

# Project root = the parent of this file's folder (src/) = the local/ project dir. All
# artifact paths below are anchored to it, so the pipeline runs the same whether you are
# inside local/ (`python run.py`) or call it from the repo root (`python local/run.py`);
# the cwd does not matter.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root(*parts):
    return os.path.join(ROOT, *parts)


def res_path(cam):
    """Absolute path of a camera's MOT result file, e.g. <root>/results/res_cam1.txt."""
    return os.path.join(RESULTS_DIR, f"res_{cam}.txt")


# --- KAFKA ---
BROKER = os.environ.get("REID_BROKER", "localhost:9092")

TOPIC_VIDEO = "video_reid_stream"
TOPIC_GALLERY = "reid_gallery_stream"
TOPIC_MATCH = "reid_matches"

# Every topic the pipeline uses. run.py pre-creates these before launching any
# component so consumers never subscribe to a not-yet-existing topic (which, with
# auto.offset.reset=latest, makes them miss messages until librdkafka's slow
# metadata refresh finally discovers the lazily-created topic).
TOPICS = (TOPIC_VIDEO, TOPIC_GALLERY, TOPIC_MATCH)

# --- CONSUMER GROUP IDS ---
GROUP_CAM1 = "cam1_service_v2"
GROUP_CAM2 = "cam2_service_v2"
GROUP_CAM3 = "cam3_service_best_shot_v1"
GROUP_VISUALIZER = "visualizer_bottom_v4"

# --- MODEL PATHS ---
YOLO_CAM1 = _root("weights", "cam1.pt")
YOLO_CAM2 = _root("weights", "cam2.pt")
YOLO_CAM3 = _root("weights", "cam1.pt")          # cam3 currently reuses cam1's weights
OSNET_WEIGHTS = _root("weights", "osnet_cam123.pth")

# --- VIDEO SOURCES (producer) ---
VIDEO_SOURCES = {
    "cam1": _root("data", "cam1_640.mp4"),
    "cam2": _root("data", "cam2_640.mp4"),
    "cam3": _root("data", "cam3_640.mp4"),
}

# --- DIRECTORIES (all anchored to ROOT) ---
LOG_DIR = os.environ.get("REID_LOG_DIR") or _root("logs")
RESULTS_DIR = _root("results")
GT_DIR = _root("gt")                # gt_cam*.txt live under local/gt/
TEMP_EVAL_DIR = _root("temp_eval_data")
