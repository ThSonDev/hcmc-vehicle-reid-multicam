import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Add path để import reid_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import reid_utils as utils
from ultralytics import YOLO
import supervision as sv

# --- 1. CONFIGURATION MATRIX (Mang config Local vào đây) ---
CAM_CONFIGS = {
    "cam1": {
        "skip_frames": 2,
        "yolo_conf": 0.50,
        "min_area": 800,
        "track_thresh": 0.2,
        "match_thresh": 0.8, # Threshold nội bộ tracker (nếu cần)
        # Cam 1 là nguồn, không cần travel_time match
    },
    "cam2": {
        "skip_frames": 2, # Mặc định xử lý hết hoặc tùy chỉnh
        "yolo_conf": 0.50,
        "min_area": 800,
        "track_thresh": 0.2,
        "similarity_threshold": 0.65, # Config riêng cam 2
        "min_travel_time": 1.0,       # Config riêng cam 2
        "max_travel_time": 10.0
    },
    "cam3": {
        "skip_frames": 2,
        "yolo_conf": 0.50,
        "min_area": 800,
        "track_thresh": 0.2,
        "similarity_threshold": 0.5,  # Config riêng cam 3 (thấp hơn cam 2 do xa)
        "min_travel_time": 20.0,      # Config riêng cam 3
        "max_travel_time": 40.0
    }
}

# --- 2. SPARK PANDAS UDF LOGIC ---

def process_stream_udf_factory(cam_id, yolo_path, osnet_path):
    """
    Factory Function: Trả về hàm UDF thực thi.
    Tham số yolo_path, osnet_path được đóng gói vào closure để gửi sang Executor.
    """
    
    # Lấy config cụ thể cho camera hiện tại
    # Nếu không tìm thấy cam_id, dùng config mặc định an toàn
    cfg = CAM_CONFIGS.get(cam_id, CAM_CONFIGS["cam1"])
    
    def process_partition(iterator):
        """
        Đây là hàm chạy trên Executor (Worker Node).
        Input: Iterator các batch (Pandas DataFrame).
        Output: Iterator các batch kết quả.
        """
        
        # --- A. HEAVY INITIALIZATION (Chạy 1 lần duy nhất mỗi Partition) ---
        print(f"[{cam_id}] 🚀 Executor Start: Init Models & DB Connection...")
        try:
            # 1. Load AI Models
            model = YOLO(yolo_path)
            extractor = utils.FeatureExtractor(model_path=osnet_path)
            
            # 2. Init Tracker với Config
            tracker = sv.ByteTrack(
                track_activation_threshold=cfg.get('track_thresh', 0.2), 
                frame_rate=30
            )
            
            # 3. Kết nối DB (Mỗi executor cần 1 connection riêng)
            conn = utils.get_db_connection()
            
        except Exception as e:
            print(f"[{cam_id}] ❌ Init Failed: {e}")
            yield pd.DataFrame() # Trả về rỗng nếu lỗi
            return

        # Buffer cục bộ cho logic "Best Shot" (nếu cần dùng)
        track_buffer = {} 

        # --- B. PROCESS BATCHES (Vòng lặp xử lý dữ liệu) ---
        for batch_df in iterator:
            # batch_df là 1 cục dữ liệu (ví dụ 100 frame) được Spark chuyển thành Pandas DF
            
            results_status = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Decode ảnh
                    nparr = np.frombuffer(row['value'], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None: 
                        results_status.append("error_decode")
                        continue
                    
                    # Lấy Timestamp thực từ header (hoặc dùng time hiện tại nếu mất)
                    # Giả sử format header đã được parse hoặc lấy time.time()
                    ts_now = time.time()

                    # 1. AI Inference (Dùng Config)
                    results = model(frame, verbose=False, conf=cfg.get('yolo_conf', 0.5))[0]
                    detections = sv.Detections.from_ultralytics(results)
                    detections = utils.merge_truck_boxes(detections, frame.shape)
                    detections = tracker.update_with_detections(detections)
                    
                    # 2. Lọc đối tượng
                    valid_crops = []
                    valid_tids = []
                    
                    for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                        tid = int(tid)
                        x1, y1, x2, y2 = map(int, xyxy)
                        area = (x2-x1)*(y2-y1)
                        
                        # Dùng config MIN_AREA
                        if area < cfg.get('min_area', 800): continue
                        
                        # Logic đơn giản: Lấy crop hiện tại
                        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                        valid_crops.append(crop)
                        valid_tids.append(tid)

                    if not valid_crops:
                        results_status.append("no_obj")
                        continue

                    # 3. Extract Features (Batch)
                    features = extractor.extract_batch(valid_crops)
                    
                    cur = conn.cursor()
                    
                    # 4. Logic Nghiệp vụ riêng từng Cam (Dùng Config Time/Threshold)
                    for i, tid in enumerate(valid_tids):
                        feat_vec = features[i]
                        img_crop = valid_crops[i]
                        
                        if cam_id == 'cam1':
                            # --- CAM 1: SOURCE ---
                            img_path = utils.save_image_static(img_crop, cam_id, tid)
                            
                            # Upsert Track
                            cur.execute("""
                                INSERT INTO vehicle_tracks (global_track_id, cam_id, track_id, timestamp, img_path, feature_vector)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (global_track_id) DO UPDATE 
                                SET img_path = EXCLUDED.img_path, feature_vector = EXCLUDED.feature_vector;
                            """, (f"{cam_id}_{tid}", cam_id, tid, ts_now, img_path, feat_vec.tolist()))
                            
                            # Init Match Row
                            cur.execute("INSERT INTO vehicle_matches (cam1_track_id) VALUES (%s) ON CONFLICT DO NOTHING;", (tid,))
                            
                        else:
                            # --- CAM 2 & 3: MATCHING ---
                            # Sử dụng MIN/MAX TRAVEL TIME từ Config
                            min_t = cfg.get('min_travel_time', 0)
                            max_t = cfg.get('max_travel_time', 999)
                            sim_thresh = cfg.get('similarity_threshold', 0.6)
                            
                            # Query thông minh: Chỉ lấy xe Cam 1 xuất hiện trong khoảng thời gian hợp lệ
                            # ts_now - max_t <= t_cam1 <= ts_now - min_t
                            # Ví dụ: Xe tới Cam 2 lúc 10:00:10. Min travel 1s, Max 10s.
                            # -> Tìm xe ở Cam 1 từ 10:00:00 đến 10:00:09.
                            
                            t_start = ts_now - max_t
                            t_end = ts_now - min_t
                            
                            query = """
                                SELECT track_id, feature_vector 
                                FROM vehicle_tracks 
                                WHERE cam_id = 'cam1' 
                                AND timestamp BETWEEN %s AND %s
                            """
                            cur.execute(query, (t_start, t_end))
                            candidates = cur.fetchall()
                            
                            best_match_id = None
                            best_score = 0.0
                            
                            if candidates:
                                c1_ids = [c[0] for c in candidates]
                                # Convert list float[] từ DB sang numpy matrix
                                c1_feats = np.array([c[1] for c in candidates]) 
                                
                                # Tính Similarity (Dot Product)
                                sims = np.dot(feat_vec, c1_feats.T)
                                best_idx = np.argmax(sims)
                                best_score = sims[best_idx]
                                
                                # Dùng Config Similarity Threshold
                                if best_score > sim_thresh:
                                    best_match_id = c1_ids[best_idx]
                            
                            # Lưu Track hiện tại
                            img_path = utils.save_image_static(img_crop, cam_id, tid)
                            cur.execute("""
                                INSERT INTO vehicle_tracks (global_track_id, cam_id, track_id, timestamp, img_path, feature_vector)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (global_track_id) DO UPDATE SET img_path = EXCLUDED.img_path;
                            """, (f"{cam_id}_{tid}", cam_id, tid, ts_now, img_path, feat_vec.tolist()))
                            
                            # Update Match nếu tìm thấy
                            if best_match_id:
                                col_track = f"{cam_id}_track_id"
                                col_score = f"{cam_id}_score"
                                # Query dynamic update
                                sql_update = f"UPDATE vehicle_matches SET {col_track} = %s, {col_score} = %s WHERE cam1_track_id = %s"
                                cur.execute(sql_update, (tid, float(best_score), best_match_id))
                                print(f"✅ [{cam_id}] Match Found! Local:{tid} <-> Cam1:{best_match_id} (Score:{best_score:.2f})")
                    
                    conn.commit()
                    cur.close()
                    results_status.append("processed")
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    results_status.append("error")

            # Yield kết quả dummy để Spark biết batch này đã xong
            yield pd.DataFrame({'status': results_status})
        
        # Đóng kết nối khi Partition xử lý xong
        if 'conn' in locals() and conn:
            conn.close()

    return process_partition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', type=str, required=True)
    parser.add_argument('--yolo', type=str, required=True)
    parser.add_argument('--osnet', type=str, required=True)
    parser.add_argument('--topic', type=str, default='video_reid_stream')
    parser.add_argument('--broker', type=str, default='kafka:9092')
    args = parser.parse_args()

    print(f"[{args.cam_id}] Starting Spark Consumer (Structured Streaming + Pandas UDF)...")
    
    # Auto Init DB
    try: utils.init_db_tables()
    except: pass

    spark = SparkSession.builder \
        .appName(f"ReID_{args.cam_id}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    # 1. Read Stream from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", args.broker) \
        .option("subscribe", args.topic) \
        .option("startingOffsets", "latest") \
        .load()

    # 2. Filter theo cam_id (để mỗi spark job chỉ xử lý 1 cam)
    df_filtered = df.filter(df.key.cast("string") == args.cam_id)

    # 3. Apply Pandas UDF (MapInPandas)
    # Đây là cách dùng Pandas UDF hiện đại (Spark 3.x) cho Scalar Iterator
    out_schema = StructType([StructField('status', StringType(), True)])
    
    # Factory pattern: Truyền config vào closure
    inference_udf = process_stream_udf_factory(
        args.cam_id, 
        args.yolo, 
        args.osnet
    )

    # mapInPandas nhận vào function (iterator -> iterator)
    query = df_filtered.mapInPandas(inference_udf, schema=out_schema) \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("checkpointLocation", f"/tmp/checkpoints/{args.cam_id}") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()