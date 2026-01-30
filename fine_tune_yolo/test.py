import os
import glob
import cv2

IMG_DIR = "dataset_cam3/images/train"
LBL_DIR = "dataset_cam3/labels/train"

FPS = 30
DELAY_MS = int(1000 / FPS)

# class names theo data.yaml của bạn (đổi cho đúng)
CLASS_NAMES = ["class0", "class1", "class2"]  # sửa lại cho khớp nc=3

def read_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cls, x, y, w, h))
    return boxes

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

def main():
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not img_paths:
        print(f"[ERROR] Không thấy ảnh trong: {IMG_DIR}")
        return

    paused = False
    idx = 0

    print("Controls: SPACE = pause/play | a = prev | d = next | q = quit")

    while True:
        if idx < 0:
            idx = 0
        if idx >= len(img_paths):
            idx = len(img_paths) - 1

        img_path = img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LBL_DIR, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Không đọc được ảnh: {img_path}")
            idx += 1
            continue

        h, w = img.shape[:2]
        boxes = read_yolo_label(label_path)

        vis = img.copy()

        for (cls, x, y, bw, bh) in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            name = str(cls)
            if 0 <= cls < len(CLASS_NAMES):
                name = CLASS_NAMES[cls]

            cv2.putText(
                vis,
                f"{name}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"{idx+1}/{len(img_paths)}  {os.path.basename(img_path)}  boxes={len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO Annotation Check (30 FPS)", vis)

        key = cv2.waitKey(0 if paused else DELAY_MS) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            idx -= 1
        elif key == ord("d"):
            idx += 1
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
