import streamlit as st
import numpy as np
import cv2
from confluent_kafka import Consumer

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Multi-Cam Monitor")

# Kafka Config
BROKER = "kafka:9092" # Lưu ý: Chạy trong Docker thì dùng tên service kafka
TOPIC = "video_reid_stream"
GROUP_ID = "streamlit_monitor_view"

@st.cache_resource
def get_consumer():
    """Tạo Kafka Consumer một lần duy nhất"""
    conf = {
        'bootstrap.servers': BROKER,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'latest' # Luôn xem data mới nhất để không bị delay
    }
    c = Consumer(conf)
    c.subscribe([TOPIC])
    return c

def main():
    st.title("📹 Live Camera Monitor (Kafka Stream)")
    
    # Tạo 3 cột layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Camera 1")
        ph1 = st.empty() # Placeholder
    
    with col2:
        st.header("Camera 2")
        ph2 = st.empty()
    
    with col3:
        st.header("Camera 3")
        ph3 = st.empty()
    
    # Nút Stop để dừng vòng lặp (vì Streamlit chạy script từ trên xuống dưới)
    stop_btn = st.button("Stop Monitoring")
    
    if stop_btn:
        st.stop()

    # Status
    status_text = st.empty()
    
    # Init Consumer
    try:
        consumer = get_consumer()
        status_text.success(f"Connected to Kafka: {BROKER}")
    except Exception as e:
        status_text.error(f"Kafka Connection Error: {e}")
        st.stop()

    # Vòng lặp đọc Kafka
    while True:
        # Đọc batch message để tăng tốc độ hiển thị
        msgs = consumer.consume(num_messages=10, timeout=0.1)
        
        if not msgs:
            continue
            
        for msg in msgs:
            if msg.error():
                continue
            
            # Key là cam id: cam1, cam2, cam3
            key = msg.key().decode('utf-8') if msg.key() else "unknown"
            
            # Decode ảnh
            nparr = np.frombuffer(msg.value(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Chuyển BGR (OpenCV) -> RGB (Streamlit/Pillow)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cập nhật vào đúng cột
            if key == "cam1":
                ph1.image(frame, channels="RGB", use_column_width=True)
            elif key == "cam2":
                ph2.image(frame, channels="RGB", use_column_width=True)
            elif key == "cam3":
                ph3.image(frame, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()