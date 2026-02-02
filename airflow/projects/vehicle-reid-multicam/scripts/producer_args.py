import cv2
import time
import json
import threading
import signal
import sys
import os
import argparse
import logging
from confluent_kafka import Producer

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Thêm path hiện tại để import reid_utils nếu cần (dù producer này ít dùng utils nhưng giữ practice tốt)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global Event để dừng thread an toàn
exit_event = threading.Event()

def signal_handler(sig, frame):
    logger.info("Signal received. Stopping producer...")
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def delivery_report(err, msg):
    """Callback báo cáo tình trạng gửi Kafka (Optional log)"""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")

def stream_worker(camera_id, video_path, producer_instance, topic, target_fps, target_width, quality, start_time_ref):
    """Worker xử lý từng camera"""
    if not os.path.exists(video_path):
        logger.error(f"[{camera_id}] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # Tính toán skip frame để đạt target fps
    frame_skip_interval = max(1, int(round(original_fps / target_fps)))
    
    logger.info(f"[{camera_id}] Streaming started. Src FPS: {original_fps:.2f} -> Target: {target_fps}. Skip: {frame_skip_interval}")

    frame_count = 0
    # Biến local để track frame gửi đi
    sent_count = 0 

    try:
        while not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.info(f"[{camera_id}] Video ended. Restarting loop.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue

            # --- Logic giả lập Real-time (Sync Time) ---
            # Tính thời điểm lý thuyết frame này phải xuất hiện
            target_ts = frame_count * (1.0 / original_fps)
            # Thời gian thực tế đã trôi qua từ lúc bắt đầu chạy script
            current_run = time.time() - start_time_ref
            wait = target_ts - current_run
            
            if wait > 0:
                time.sleep(wait)

            # Skip frame nếu cần giảm FPS
            if frame_count % frame_skip_interval != 0:
                frame_count += 1
                continue

            # Resize
            if target_width is not None and target_width > 0:
                h, w = frame.shape[:2]
                if w != target_width:
                    scale = target_width / w
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # Encode JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Timestamp gửi đi là thời gian thực hiện tại
            meta = {
                "timestamp": time.time(),
                "frame_idx": sent_count,
                "original_frame_idx": frame_count
            }
            
            try:
                producer_instance.produce(
                    topic,
                    key=camera_id.encode('utf-8'),
                    value=buffer.tobytes(),
                    headers=[("meta", json.dumps(meta).encode('utf-8'))],
                    on_delivery=delivery_report
                )
                # Poll nhẹ để Kafka client xử lý callback
                producer_instance.poll(0)
            except BufferError:
                logger.warning(f"[{camera_id}] Kafka Buffer full, waiting...")
                producer_instance.poll(0.1)

            frame_count += 1
            sent_count += 1

    except Exception as e:
        logger.error(f"[{camera_id}] Exception: {e}")
    finally:
        cap.release()
        logger.info(f"[{camera_id}] Worker stopped.")

def main():
    parser = argparse.ArgumentParser(description="Multi-camera Kafka Producer")
    
    # Args cho path video
    parser.add_argument('--cam1', type=str, required=True, help='Path to video file for Cam 1')
    parser.add_argument('--cam2', type=str, required=True, help='Path to video file for Cam 2')
    parser.add_argument('--cam3', type=str, required=True, help='Path to video file for Cam 3')
    
    # Args cấu hình
    parser.add_argument('--broker', type=str, default="kafka:9092", help='Kafka Broker Address')
    parser.add_argument('--topic', type=str, default="video_reid_stream", help='Kafka Topic')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--width', type=int, default=0, help='Target Width (0 = original)')
    parser.add_argument('--quality', type=int, default=70, help='JPEG Quality (0-100)')
    parser.add_argument('--duration', type=int, default=0, help='Duration to run in seconds (0 = forever)')

    args = parser.parse_args()

    # Cấu hình Producer
    conf = {
        'bootstrap.servers': args.broker,
        'queue.buffering.max.messages': 5000,
        'linger.ms': 10,
        'compression.type': 'lz4'
    }
    
    logger.info(f"Init Kafka Producer to {args.broker} on topic {args.topic}")
    producer = Producer(conf)

    # Dictionary map cam_id -> path
    video_sources = {
        "cam1": args.cam1,
        "cam2": args.cam2,
        "cam3": args.cam3
    }

    threads = []
    # Thời gian gốc để đồng bộ 3 cam
    global_start = time.time()

    # Khởi động threads
    for cam_id, path in video_sources.items():
        t = threading.Thread(
            target=stream_worker, 
            args=(cam_id, path, producer, args.topic, args.fps, args.width, args.quality, global_start)
        )
        t.daemon = True 
        t.start()
        threads.append(t)

    logger.info("All camera threads started.")

    try:
        # Loop chính để giữ chương trình chạy
        start_wait = time.time()
        while not exit_event.is_set():
            time.sleep(1)
            # Kiểm tra nếu tất cả thread chết thì dừng
            if not any(t.is_alive() for t in threads):
                logger.warning("All threads died. Exiting.")
                break
            
            # Nếu set duration, kiểm tra thời gian
            if args.duration > 0 and (time.time() - start_wait > args.duration):
                logger.info(f"Duration {args.duration}s reached. Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        # 1. Báo hiệu cho các thread dừng lại
        exit_event.set()
        
        # 2. Đợi các thread con kết thúc hẳn (QUAN TRỌNG ĐỂ TRÁNH CRASH)
        for t in threads:
            if t.is_alive():
                t.join(timeout=2.0) # Đợi tối đa 2s mỗi thread

        # 3. Sau khi thread dừng, mới flush kafka
        logger.info("Flushing Kafka messages...")
        producer.flush(timeout=5.0)
        
        logger.info("Cleanup done. Exiting.")
        # 4. Ép buộc thoát sạch sẽ để tránh lỗi C++ destructor
        sys.exit(0)

if __name__ == "__main__":
    main()