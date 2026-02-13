from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime
import os
import torch

# --- CONFIG ---
# Path trong container Airflow
BASE_DIR = "/opt/airflow/projects/vehicle-reid-multicam"
SCRIPT_PATH = f"{BASE_DIR}/scripts/spark_consumer.py"

MODELS = {
    "yolo_cam1": "/opt/airflow/models/cam1.pt",
    "yolo_cam2": "/opt/airflow/models/cam2.pt",
    "osnet": "/opt/airflow/models/osnet_cam123.pth"
}

def check_requirements():
    print(">>> Checking Environment...")
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected. Running on CPU (Will be slow).")

    # Check Models
    for name, path in MODELS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {name} at {path}")
        print(f"✅ Found {name}")

    # Check Script
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError(f"Missing consumer script: {SCRIPT_PATH}")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'vehicle_reid_consumers',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['spark', 'reid']
) as dag:

    t1_check = PythonOperator(
        task_id='check_resources',
        python_callable=check_requirements
    )

    # Helper function to create Spark Task
    def create_spark_task(cam_id, yolo_model):
        utils_path = f"{BASE_DIR}/scripts/reid_utils.py"
        
        cmd = f"""
        spark-submit \
        --master local[*] \
        --py-files {utils_path} \
        {SCRIPT_PATH} \
        --cam_id {cam_id} \
        --yolo {yolo_model} \
        --osnet {MODELS['osnet']} \
        --broker kafka:9092
        """
        return BashOperator(
            task_id=f'run_{cam_id}_consumer',
            bash_command=cmd
        )

    t_cam1 = create_spark_task('cam1', MODELS['yolo_cam1'])
    t_cam2 = create_spark_task('cam2', MODELS['yolo_cam2'])
    # Cam 3 dùng chung model Cam 1 (như yêu cầu)
    t_cam3 = create_spark_task('cam3', MODELS['yolo_cam1']) 

    t_cleanup = BashOperator(
        task_id='cleanup_temp',
        bash_command='rm -rf /tmp/checkpoints/*' # Xóa checkpoint spark cũ để tránh lỗi schema
    )

    t1_check >> t_cleanup >> [t_cam1, t_cam2, t_cam3]