import cv2
import numpy as np

VIDEO_PATH = r"./data/cam1_half.mp4"  # change this
MOT_PATH = r"gt_cam1.txt"  # change this
# Optional — class names based on your class_id mapping
CLASS_NAMES = {
    1: "car",
    2: "truck",
    3: "bus"
}

# Class-based colors (BGR format)
CLASS_COLORS = {
    1: (0, 255, 0),    # car = green
    3: (255, 0, 0),    # bus = blue
    2: (0, 0, 255)     # truck = red
}

# LOAD MOT ANNOTATIONS
tracks_by_frame = {}

with open(MOT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue  # skip invalid lines

        frame = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = map(float, parts[2:6])
        class_id = int(parts[7])

        if frame not in tracks_by_frame:
            tracks_by_frame[frame] = []

        tracks_by_frame[frame].append({
            "track_id": track_id,
            "bbox": (int(x), int(y), int(w), int(h)),
            "class_id": class_id
        })

print(f"Loaded {len(tracks_by_frame)} frames of annotations")

# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video opened ({fps:.1f} FPS, {total_frames} frames)")

frame_idx = 0
paused = False

cv2.namedWindow("MOT Viewer", cv2.WINDOW_NORMAL)

# DISPLAY VIDEO WITH TRACKS
while True:

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

    # Draw bounding boxes
    if frame_idx in tracks_by_frame:
        for obj in tracks_by_frame[frame_idx]:
            x, y, w, h = obj["bbox"]
            track_id = obj["track_id"]
            class_id = obj["class_id"]

            label = CLASS_NAMES.get(class_id, str(class_id))
            color = CLASS_COLORS.get(class_id, (0, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} {track_id}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # TIMESTAMP overlay
    seconds = frame_idx / fps
    time_str = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    cv2.putText(
        frame,
        time_str,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # AUTO-RESIZE TO FIT SCREEN
    h, w = frame.shape[:2]
    max_w, max_h = 1280, 720
    scale = min(max_w / w, max_h / h, 1.0)
    frame_display = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("MOT Viewer", frame_display)

    # KEYBOARD CONTROLS
    delay = max(1, int(1000 / fps))   # Real-time playback
    key = cv2.waitKey(delay if not paused else 0) & 0xFF

    if key == ord('q'):
        break

    # pause / resume
    if key == ord(' '):
        paused = not paused

    # rewind 2 seconds
    if key == ord('a'):
        rewind_frames = int(fps * 2)
        frame_idx = max(1, frame_idx - rewind_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # fast-forward 2 seconds
    if key == ord('d'):
        forward_frames = int(fps * 2)
        frame_idx = min(total_frames - 1, frame_idx + forward_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

cap.release()
cv2.destroyAllWindows()

print("Finished")
