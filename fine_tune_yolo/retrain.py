import os
import sys

# CONFIG

MODEL = "yolov8s.pt"

DATA_CAM2 = "dataset_cam2/data.yaml"

# train params
IMGSZ = 640          # FullHD + xe nhỏ => nên 960 (640 vẫn được nhưng dễ miss xe nhỏ)
EPOCHS = 15
BATCH = 16            # nếu out-of-memory => giảm 4
DEVICE = 0           # 0 = GPU0, "cpu" = chạy CPU (rất chậm)
WORKERS = 4

# output folders
PROJECT_CAM2 = "runs_cam2"

NAME_CAM2 = "train"

# UTILS

def check_exists(path):
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy: {path}")
        sys.exit(1)

def train_one(data_yaml, project, name):
    from ultralytics import YOLO

    print("\n==============================")
    print(f"Training with data: {data_yaml}")
    print(f"Project: {project}, Name: {name}")
    print("==============================\n")

    model = YOLO(MODEL)

    results = model.train(
        data=data_yaml,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=project,
        name=name,
        pretrained=True,
        exist_ok=True,
        verbose=True
    )

    return results

# MAIN

if __name__ == "__main__":
    check_exists(DATA_CAM2)

    # train cam2
    train_one(DATA_CAM2, PROJECT_CAM2, NAME_CAM2)

    print("\nDONE.")
    print("Cam2 best.pt:", os.path.join(PROJECT_CAM2, NAME_CAM2, "weights", "best.pt"))
