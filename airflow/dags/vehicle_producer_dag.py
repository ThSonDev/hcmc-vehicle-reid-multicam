from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os

# --- CONFIG ---
# Định nghĩa các giá trị mặc định nếu Variable chưa set
DEFAULT_CONF = {
    "CAM1_PATH": "/opt/airflow/projects/vehicle-reid-multicam/data/cam1.mp4",
    "CAM2_PATH": "/opt/airflow/projects/vehicle-reid-multicam/data/cam2.mp4",
    "CAM3_PATH": "/opt/airflow/projects/vehicle-reid-multicam/data/cam3.mp4",
    "TARGET_FPS": "30",
    "TARGET_WIDTH": "640",
    "JPEG_QUALITY": "70",
    "KAFKA_BROKER": "kafka:9092",
    "RUN_DURATION": "60" # Chạy 5 phút rồi tự tắt để tránh treo task mãi mãi
}

def check_video_paths(**kwargs):
    """Python Operator: Kiểm tra xem file video có tồn tại không"""
    # Lấy path từ kwargs hoặc Airflow Variable
    paths = {
        "cam1": kwargs.get('cam1'),
        "cam2": kwargs.get('cam2'),
        "cam3": kwargs.get('cam3')
    }
    
    missing_files = []
    print("[INFO] Checking video file paths...")
    
    for cam, path in paths.items():
        if not os.path.exists(path):
            print(f"[ERROR] {cam} path does not exist: {path}")
            missing_files.append(path)
        else:
            print(f"[INFO] {cam} OK: {path}")
            
    if missing_files:
        raise FileNotFoundError(f"Missing video files: {missing_files}")
    
    print("[INFO] All video files verified.")

# --- DAG DEFINITION ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 2),
    'email_on_failure': False,
    'retries': 0,
}

with DAG(
    'vehicle_streaming_producer',
    default_args=default_args,
    description='Stream video from files to Kafka simulate Cameras',
    schedule_interval=None, # Trigger thủ công
    catchup=False,
    tags=['reid', 'streaming'],
) as dag:

    # 1. Lấy Config từ Variable (Nếu không có dùng mặc định)
    # Bạn có thể vào Admin -> Variables để set các key này
    cam1 = Variable.get("REID_CAM1_PATH", default_var=DEFAULT_CONF["CAM1_PATH"])
    cam2 = Variable.get("REID_CAM2_PATH", default_var=DEFAULT_CONF["CAM2_PATH"])
    cam3 = Variable.get("REID_CAM3_PATH", default_var=DEFAULT_CONF["CAM3_PATH"])
    
    fps = Variable.get("REID_FPS", default_var=DEFAULT_CONF["TARGET_FPS"])
    width = Variable.get("REID_WIDTH", default_var=DEFAULT_CONF["TARGET_WIDTH"])
    quality = Variable.get("REID_QUALITY", default_var=DEFAULT_CONF["JPEG_QUALITY"])
    duration = Variable.get("REID_DURATION", default_var=DEFAULT_CONF["RUN_DURATION"])

    # 2. Task Check File
    t1_check_files = PythonOperator(
        task_id='check_video_sources',
        python_callable=check_video_paths,
        op_kwargs={
            'cam1': cam1, 
            'cam2': cam2, 
            'cam3': cam3
        }
    )

    # 3. Task Run Producer
    # Script nằm ở projects/vehicle-reid-multicam/scripts/producer_args.py
    # Lưu ý: Kafka trong mạng Docker gọi là 'kafka', không phải 'localhost'
    
    producer_cmd = f"""
    python /opt/airflow/projects/vehicle-reid-multicam/scripts/producer_args.py \
    --cam1 {cam1} \
    --cam2 {cam2} \
    --cam3 {cam3} \
    --fps {fps} \
    --width {width} \
    --quality {quality} \
    --broker kafka:9092 \
    --topic video_reid_stream \
    --duration {duration}
    """

    t2_run_producer = BashOperator(
        task_id='run_streaming_producer',
        bash_command=producer_cmd
    )

    # Flow
    t1_check_files >> t2_run_producer