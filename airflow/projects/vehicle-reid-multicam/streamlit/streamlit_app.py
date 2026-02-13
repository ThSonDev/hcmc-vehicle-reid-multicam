import streamlit as st
import pandas as pd
import psycopg2
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from confluent_kafka import Consumer

# --- CẤU HÌNH CHO PHÉP ẢNH LỖI ---
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Vehicle ReID Monitor")

DB_CONFIG = {
    "host": "postgres",
    "database": "airflow",
    "user": "airflow",
    "password": "airflow"
}

KAFKA_BROKER = "kafka:9092"
TOPIC_VIDEO = "video_reid_stream"
STATIC_ROOT = "/opt/airflow/projects/vehicle-reid-multicam/static"

# CSS Customization
st.markdown("""
<style>
    .stImage img { border-radius: 5px; }
    h3 { padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db_conn():
    """Tạo kết nối DB (Cached)"""
    return psycopg2.connect(**DB_CONFIG)

@st.cache_resource
def get_kafka_consumer():
    """Tạo Kafka Consumer (Cached)"""
    try:
        conf = {
            'bootstrap.servers': KAFKA_BROKER,
            'group.id': 'streamlit_viewer_robust_v2',
            'auto.offset.reset': 'latest'
        }
        c = Consumer(conf)
        c.subscribe([TOPIC_VIDEO])
        return c
    except Exception as e:
        return None

def load_image_robust(rel_path):
    """Hàm đọc ảnh chịu lỗi (Retry logic)"""
    if not rel_path: return None
    full_path = os.path.join(STATIC_ROOT, rel_path)
    
    # Thử đọc 3 lần, mỗi lần cách nhau 0.1s
    for _ in range(3):
        if os.path.exists(full_path):
            try:
                # Mở và ép load dữ liệu ngay lập tức
                img = Image.open(full_path)
                img.load() 
                return img
            except Exception:
                time.sleep(0.1) # Chờ Spark ghi xong
    
    # Nếu vẫn lỗi, trả về ảnh giữ chỗ hoặc None
    return None

def main():
    st.title("🚦 Traffic Surveillance & ReID System")

    # --- LAYOUT TRÊN: LIVE CAMS ---
    st.subheader("📡 Live Camera Feeds")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown("**Camera 1 (Source)**")
        ph1 = st.empty()
    with c2: 
        st.markdown("**Camera 2 (Checkpoint)**")
        ph2 = st.empty()
    with c3: 
        st.markdown("**Camera 3 (Exit)**")
        ph3 = st.empty()

    st.divider()

    # --- LAYOUT DƯỚI: ANALYTICS ---
    col_match, col_new = st.columns([2, 1])
    
    with col_match:
        st.subheader("✅ Tracking Flow (Cam1 -> Cam2 -> Cam3)")
        match_container = st.empty()
    
    with col_new:
        st.subheader("⚠️ New/Unknown at Cam 3")
        new_cam3_container = st.empty()

    # Init Resource
    consumer = get_kafka_consumer()
    conn = None
    try:
        conn = get_db_conn()
    except:
        st.error("Waiting for Database...")

    # --- MAIN LOOP (KHÔNG BAO GIỜ CHẾT) ---
    frame_count = 0
    
    while True:
        try:
            # 1. KAFKA POLLING (Nếu consumer lỗi thì bỏ qua)
            if consumer:
                msgs = consumer.consume(num_messages=5, timeout=0.02)
                for msg in msgs:
                    if msg.error(): continue
                    
                    nparr = np.frombuffer(msg.value(), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None: continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    key = msg.key().decode('utf-8') if msg.key() else ""
                    
                    if key == "cam1": ph1.image(frame, use_column_width=True)
                    elif key == "cam2": ph2.image(frame, use_column_width=True)
                    elif key == "cam3": ph3.image(frame, use_column_width=True)

            # 2. DB QUERY (Chạy mỗi 30 frame ~ 1s)
            frame_count += 1
            if frame_count % 30 == 0:
                if conn is None or conn.closed:
                    try: conn = get_db_conn()
                    except: continue # Skip nếu chưa có DB

                # Dùng cursor trong block try để handle transaction
                try:
                    with conn.cursor() as cur:
                        # A. QUERY MATCHES
                        cur.execute("""
                            SELECT 
                                m.cam1_track_id, t1.img_path,
                                m.cam2_track_id, t2.img_path, m.cam2_score,
                                m.cam3_track_id, t3.img_path, m.cam3_score
                            FROM vehicle_matches m
                            LEFT JOIN vehicle_tracks t1 ON m.cam1_track_id = t1.track_id AND t1.cam_id='cam1'
                            LEFT JOIN vehicle_tracks t2 ON m.cam2_track_id = t2.track_id AND t2.cam_id='cam2'
                            LEFT JOIN vehicle_tracks t3 ON m.cam3_track_id = t3.track_id AND t3.cam_id='cam3'
                            ORDER BY m.last_update DESC LIMIT 5
                        """)
                        matches = cur.fetchall()

                        # B. QUERY NEW CAM 3
                        cur.execute("""
                            SELECT track_id, img_path, timestamp 
                            FROM vehicle_tracks 
                            WHERE cam_id = 'cam3' 
                            AND track_id NOT IN (SELECT cam3_track_id FROM vehicle_matches WHERE cam3_track_id IS NOT NULL)
                            ORDER BY timestamp DESC LIMIT 3
                        """)
                        new_objs = cur.fetchall()
                        
                        # C. COMMIT (Quan trọng để chốt transaction)
                        conn.commit()

                        # --- RENDER UI (Chỉ render khi có dữ liệu) ---
                        with match_container.container():
                            if not matches: st.info("System initializing / No vehicles...")
                            for row in matches:
                                mc1, mc2, mc3 = st.columns(3)
                                with mc1: 
                                    if row[1]: st.image(load_image_robust(row[1]), caption=f"ID: {row[0]}")
                                with mc2:
                                    if row[2]: st.image(load_image_robust(row[3]), caption=f"Match: {row[2]} ({row[4]:.2f})")
                                    else: st.write("...")
                                with mc3:
                                    if row[5]: st.image(load_image_robust(row[6]), caption=f"Match: {row[5]} ({row[7]:.2f})")
                                    else: st.write("...")
                                st.write("---")

                        with new_cam3_container.container():
                            if not new_objs: st.info("No unknown vehicles")
                            for obj in new_objs:
                                st.image(load_image_robust(obj[1]), caption=f"Unknown ID: {obj[0]}")

                except psycopg2.Error as e:
                    # [QUAN TRỌNG] Rollback nếu lỗi SQL (bảng chưa có, query sai...)
                    conn.rollback()
                    # print(f"DB Error (Will retry): {e}") # Debug only

        except Exception as e:
            # Catch-all để app không bao giờ crash
            # print(f"App Loop Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()