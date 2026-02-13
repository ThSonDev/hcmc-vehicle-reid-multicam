import os
import shutil
import re
import yaml
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
from ultralytics import YOLO

# --- CONFIGURATION ---
PROJECT_DIR = "/opt/airflow/projects/vehicle-reid-multicam"
DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = "/opt/airflow/models"
BASE_MODEL_PATH = f"{MODELS_DIR}/yolov8n.pt" # Hoặc cam1.pt cũ

# Train Params
YOLO_EPOCHS = 50
YOLO_BATCH = 16
YOLO_IMGSZ = 640
YOLO_PATIENCE = 10
CAMERAS = ["cam1", "cam2", "cam3"]

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

# --- HELPER FUNCTIONS ---

def check_data_existence(camera_name):
    """Operator 1: Kiểm tra folder data có tồn tại không"""
    path = os.path.join(DATA_DIR, f"refinetuneyolo{camera_name}")
    if not os.path.exists(path):
        raise AirflowException(f"Data folder not found: {path}")
    print(f"Verified data path: {path}")
    return path

def get_next_version(camera_name, **kwargs):
    """Operator 2: Tính toán version tiếp theo và đẩy vào XCom"""
    cam_model_dir = os.path.join(MODELS_DIR, camera_name)
    if not os.path.exists(cam_model_dir):
        os.makedirs(cam_model_dir)
        next_ver = "ver01"
    else:
        subdirs = [d for d in os.listdir(cam_model_dir) if os.path.isdir(os.path.join(cam_model_dir, d))]
        versions = []
        for d in subdirs:
            match = re.search(r'ver(\d+)', d)
            if match:
                versions.append(int(match.group(1)))
        
        if not versions:
            next_ver = "ver01"
        else:
            next_ver = f"ver{max(versions) + 1:02d}"
    
    print(f"Next version for {camera_name}: {next_ver}")
    # Push version to XCom để các task sau dùng
    kwargs['ti'].xcom_push(key=f'{camera_name}_version', value=next_ver)
    return next_ver

def validate_dataset_structure(camera_name):
    """Operator 3: Validate cấu trúc bên trong (YAML, Images count)"""
    data_path = os.path.join(DATA_DIR, f"refinetuneyolo{camera_name}")
    yaml_file = os.path.join(data_path, "data.yaml")
    
    if not os.path.exists(yaml_file):
        raise AirflowException(f"Missing data.yaml in {data_path}")
    
    # Đọc yaml để check đường dẫn ảnh
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Dataset config valid: {config}")
    # Có thể thêm logic đếm số file ảnh ở đây nếu cần kỹ hơn

def train_yolo_logic(camera_name, **kwargs):
    """Operator 4: Thực hiện Training"""
    # Pull version từ Task trước
    version = kwargs['ti'].xcom_pull(key=f'{camera_name}_version', task_ids=f'{camera_name}_group.get_version')
    data_path = os.path.join(DATA_DIR, f"refinetuneyolo{camera_name}")
    yaml_path = os.path.join(data_path, "data.yaml")
    
    print(f"Training {camera_name} - {version}")
    
    # Load Model (Ưu tiên model cũ nhất của cam đó, nếu không thì base)
    model = YOLO(BASE_MODEL_PATH) 
    
    # Temp output directory
    project_path = os.path.join(PROJECT_DIR, "runs", "train")
    run_name = f"{camera_name}_{version}"
    
    results = model.train(
        data=yaml_path,
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH,
        imgsz=YOLO_IMGSZ,
        patience=YOLO_PATIENCE,
        project=project_path,
        name=run_name,
        exist_ok=True,
        verbose=True
    )
    # Trả về đường dẫn file best.pt cho task sau
    return os.path.join(project_path, run_name, "weights", "best.pt")

def save_and_cleanup(camera_name, **kwargs):
    """Operator 5: Move model ra folder chính thức"""
    version = kwargs['ti'].xcom_pull(key=f'{camera_name}_version', task_ids=f'{camera_name}_group.get_version')
    # Pull đường dẫn best.pt từ task train
    best_pt_path = kwargs['ti'].xcom_pull(task_ids=f'{camera_name}_group.train_yolo')
    
    final_dir = os.path.join(MODELS_DIR, camera_name, f"{camera_name}_{version}")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        
    if os.path.exists(best_pt_path):
        target_path = os.path.join(final_dir, "best.pt")
        shutil.copy(best_pt_path, target_path)
        print(f"✅ Model saved successfully to: {target_path}")
    else:
        raise AirflowException(f"Training failed to produce best.pt at {best_pt_path}")

# --- DAG DEFINITION ---

with DAG(
    'yolo_refinetune_detailed',
    default_args=default_args,
    description='Detailed YOLO Training Pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['yolo', 'granular']
) as dag:

    for cam in CAMERAS:
        with TaskGroup(group_id=f'{cam}_group') as cam_group:
            
            t1_check = PythonOperator(
                task_id='check_data_exists',
                python_callable=check_data_existence,
                op_kwargs={'camera_name': cam}
            )
            
            t2_ver = PythonOperator(
                task_id='get_version',
                python_callable=get_next_version,
                op_kwargs={'camera_name': cam}
            )
            
            t3_val = PythonOperator(
                task_id='validate_structure',
                python_callable=validate_dataset_structure,
                op_kwargs={'camera_name': cam}
            )
            
            t4_train = PythonOperator(
                task_id='train_yolo',
                python_callable=train_yolo_logic,
                op_kwargs={'camera_name': cam}
            )
            
            t5_save = PythonOperator(
                task_id='save_model',
                python_callable=save_and_cleanup,
                op_kwargs={'camera_name': cam}
            )

            # Define flow
            t1_check >> t2_ver >> t3_val >> t4_train >> t5_save