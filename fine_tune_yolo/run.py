import os
import cv2

# CONFIG (SỬA ĐƯỜNG DẪN Ở ĐÂY)

CAM1_VIDEO = "../data/cam1_half.mp4"
CAM2_VIDEO = "../data/cam2_half.mp4"
CAM1_MOT = "gt_cam1.txt"
CAM2_MOT = "gt_cam2.txt"

CAM3_VIDEO = "../data/cam3_half.mp4"
CAM3_MOT = "../gt_cam3.txt"

# class_id trong MOT -> tên class
CLASS_NAMES = {
    1: "car",
    2: "truck",
    3: "bus"
}

# YOLO class order (0..N-1)
YOLO_NAMES = ["car", "truck", "bus"]
NAME_TO_YOLO_ID = {name: i for i, name in enumerate(YOLO_NAMES)}

# Filter box (tuỳ chọn)
VISIBILITY_THRES = 0.0     # nếu muốn lọc box mờ: đặt 0.2 hoặc 0.3
SKIP_IGNORED = True        # bỏ not_ignored == 0

# Image output
IMG_EXT = ".jpg"
JPEG_QUALITY = 95          # 0..100

# UTILS

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_mot_file(mot_path):
    """
    Returns:
        dict: frame_id -> list of (class_id, x, y, w, h, not_ignored, visibility)
    """
    frames = {}
    if not os.path.isfile(mot_path):
        raise FileNotFoundError(f"Không tìm thấy MOT file: {mot_path}")

    with open(mot_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 9:
                # nếu file bạn có thêm cột thì vẫn ok, nhưng thiếu thì skip
                continue

            frame_id = int(float(parts[0]))
            # track_id = int(float(parts[1]))  # không cần cho YOLO
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            not_ignored = int(float(parts[6]))
            class_id = int(float(parts[7]))
            visibility = float(parts[8])

            frames.setdefault(frame_id, []).append((class_id, x, y, w, h, not_ignored, visibility))

    return frames

def mot_box_to_yolo_line(class_id, x, y, w, h, img_w, img_h):
    """
    MOT: x,y,w,h top-left pixel
    YOLO: cls x_center y_center w h normalized
    """
    if class_id not in CLASS_NAMES:
        return None

    cls_name = CLASS_NAMES[class_id]
    if cls_name not in NAME_TO_YOLO_ID:
        return None

    yolo_cls = NAME_TO_YOLO_ID[cls_name]

    cx = x + w / 2.0
    cy = y + h / 2.0

    # normalize
    x_center = cx / img_w
    y_center = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # clamp (tránh lỗi vượt biên)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    # bỏ box quá nhỏ hoặc sai
    if w_norm <= 0 or h_norm <= 0:
        return None

    return f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def export_dataset(video_path, mot_path, out_root, prefix):
    """
    Tạo dataset YOLO train-only:
        out_root/images/train/prefix_000001.jpg
        out_root/labels/train/prefix_000001.txt
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    ensure_dir(out_root)
    images_dir = os.path.join(out_root, "images", "train")
    labels_dir = os.path.join(out_root, "labels", "train")
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    mot_frames = read_mot_file(mot_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n=== Export {prefix} ===")
    print(f"Video: {video_path}")
    print(f"MOT:   {mot_path}")
    print(f"Size:  {img_w}x{img_h}, total_frames={total_frames}")
    print(f"Out:   {out_root}")

    frame_idx = 0
    written_images = 0
    written_labels = 0

    # OpenCV đọc frame theo thứ tự 1..N, còn frame_idx mình tự tăng
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # tên file theo frame_id (6 digits)
        base_name = f"{prefix}_{frame_idx:06d}"
        img_path = os.path.join(images_dir, base_name + IMG_EXT)
        label_path = os.path.join(labels_dir, base_name + ".txt")

        # luôn ghi ảnh ra (để dataset đủ frame)
        # Nếu bạn muốn chỉ ghi frame có bbox thì có thể đổi logic
        if IMG_EXT.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        else:
            cv2.imwrite(img_path, frame)
        written_images += 1

        # tạo label file (có thể rỗng nếu không có object)
        lines = []
        if frame_idx in mot_frames:
            for (class_id, x, y, w, h, not_ignored, visibility) in mot_frames[frame_idx]:
                if SKIP_IGNORED and not_ignored == 0:
                    continue
                if visibility < VISIBILITY_THRES:
                    continue

                yolo_line = mot_box_to_yolo_line(class_id, x, y, w, h, img_w, img_h)
                if yolo_line is not None:
                    lines.append(yolo_line)

        # YOLO chấp nhận label file rỗng (ảnh không có object)
        with open(label_path, "w") as f:
            if lines:
                f.write("\n".join(lines))
        written_labels += 1

        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    print(f"Done {prefix}: images={written_images}, labels={written_labels}")

def write_data_yaml(out_root):
    """
    Viết data.yaml trong mỗi dataset folder để train YOLO tiện hơn.
    """
    yaml_path = os.path.join(out_root, "data.yaml")
    text = f"""path: {out_root}
train: images/train
val: images/train

names:
  0: car
  1: truck
  2: bus
"""
    with open(yaml_path, "w") as f:
        f.write(text)

# MAIN

if __name__ == "__main__":
    # export_dataset(CAM1_VIDEO, CAM1_MOT, "dataset_cam1", "cam1")
    # write_data_yaml("dataset_cam1")

    # export_dataset(CAM2_VIDEO, CAM2_MOT, "dataset_cam2", "cam2")
    # write_data_yaml("dataset_cam2")
    export_dataset(CAM3_VIDEO, CAM3_MOT, "dataset_cam3", "cam3")
    write_data_yaml("dataset_cam3")

    print("\nAll done.")
    print("Bạn có thể train YOLOv8 ví dụ:")
    print("  yolo detect train model=yolov8s.pt data=dataset_cam1/data.yaml imgsz=640 epochs=80 batch=16")
    print("  yolo detect train model=yolov8s.pt data=dataset_cam2/data.yaml imgsz=640 epochs=80 batch=16")
