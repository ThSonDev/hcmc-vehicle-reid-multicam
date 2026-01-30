import cv2
import time
import json
import threading
import signal
import sys
import os
from confluent_kafka import Producer

# --- CẤU HÌNH ---
BROKER = "localhost:9092"
TOPIC = "video_reid_stream"
VIDEO_SOURCES = {
    "cam1": "data/cam1_640.mp4",
    "cam2": "data/cam2_640.mp4",
    "cam3": "data/cam3_640.mp4" # Thêm Cam 3
}

# --- CẤU HÌNH TỐI ƯU ---
TARGET_WIDTH = None 
TARGET_FPS = 30      
JPEG_QUALITY = 70    

exit_event = threading.Event()

def signal_handler(sig, frame):
    print("\n[System] Stopping...")
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)

def delivery_report(err, msg): pass 

def stream_worker(camera_id, video_path, producer, start_time_ref):
    if not os.path.exists(video_path):
        print(f"[{camera_id}] Error: Not found {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip_interval = max(1, int(round(original_fps / TARGET_FPS)))
    
    print(f"[{camera_id}] Streaming {TARGET_FPS}fps...")

    frame_count = 0
    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Replay")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue

        # Real-time pacing
        target_ts = frame_count * (1.0 / original_fps)
        current_run = time.time() - start_time_ref
        wait = target_ts - current_run
        if wait > 0: time.sleep(wait)

        if frame_count % frame_skip_interval != 0:
            frame_count += 1
            continue

        if TARGET_WIDTH is not None:
            h, w = frame.shape[:2]
            if w != TARGET_WIDTH:
                scale = TARGET_WIDTH / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        meta = { "timestamp": time.time(), "frame_idx": frame_count }
        
        try:
            producer.produce(
                TOPIC,
                key=camera_id.encode('utf-8'),
                value=buffer.tobytes(),
                headers=[("meta", json.dumps(meta).encode('utf-8'))],
                on_delivery=delivery_report
            )
            producer.poll(0)
        except BufferError:
            producer.poll(0.1)

        frame_count += 1
    cap.release()

if __name__ == "__main__":
    conf = {
        'bootstrap.servers': BROKER,
        'queue.buffering.max.messages': 50000,
        'linger.ms': 10,
        'compression.type': 'lz4'
    }
    producer = Producer(conf)
    threads = []
    global_start = time.time()

    for cam_id, path in VIDEO_SOURCES.items():
        t = threading.Thread(target=stream_worker, args=(cam_id, path, producer, global_start))
        t.daemon = True 
        t.start()
        threads.append(t)

    try:
        while not exit_event.is_set():
            time.sleep(1)
            if not any(t.is_alive() for t in threads): break
    except KeyboardInterrupt: pass
    finally:
        exit_event.set()
        producer.flush(timeout=5.0)