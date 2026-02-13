import os
import shutil
import re
import cv2
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException

# Import Torchreid (đã có trong Dockerfile)
try:
    from torchreid import data, models, optim, engine
    import torchreid
except ImportError:
    pass # Sẽ báo lỗi runtime nếu thiếu

# --- CONFIGURATION ---
PROJECT_DIR = "/opt/airflow/projects/vehicle-reid-multicam"
DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = "/opt/airflow/models"
TEMP_ROOT = f"{PROJECT_DIR}/temp_reid_process"

# ReID Params
BASE_MODEL_NAME = "osnet_x1_0"
MAX_EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0003
TRAIN_RATIO = 0.8

# Data Source Mapping
VIDEO_MAP = {"cam1": "cam1.mp4", "cam2": "cam2.mp4", "cam3": "cam3.mp4"}
GT_MAP = {"cam1": "gt_cam1.txt", "cam2": "gt_cam2.txt", "cam3": "gt_cam3.txt"}

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# --- OPERATOR FUNCTIONS ---

def check_resources():
    """Step 1: Check inputs"""
    for cam, vname in VIDEO_MAP.items():
        vpath = os.path.join(DATA_DIR, vname)
        gtpath = os.path.join(DATA_DIR, GT_MAP[cam])
        if not os.path.exists(vpath): raise AirflowException(f"Missing {vpath}")
        if not os.path.exists(gtpath): raise AirflowException(f"Missing {gtpath}")
    print("All video and GT files found.")

def calculate_version(**kwargs):
    """Step 2: Get next osnet version"""
    base_dir = os.path.join(MODELS_DIR, "osnet")
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    
    versions = []
    for d in os.listdir(base_dir):
        match = re.search(r'osnet_ver(\d+)', d)
        if match: versions.append(int(match.group(1)))
    
    next_ver = f"ver{max(versions) + 1:02d}" if versions else "ver01"
    print(f"Target Version: {next_ver}")
    kwargs['ti'].xcom_push(key='osnet_version', value=next_ver)

def extract_frames_from_video(**kwargs):
    """Step 3: Extract & Crop images (Heavy Task)"""
    # Tạo folder tạm
    train_dir = os.path.join(TEMP_ROOT, "bounding_box_train")
    query_dir = os.path.join(TEMP_ROOT, "query")
    gallery_dir = os.path.join(TEMP_ROOT, "bounding_box_test")
    
    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(train_dir); os.makedirs(query_dir); os.makedirs(gallery_dir)
    
    # 3.1 Gather IDs
    all_ids = []
    for cam in GT_MAP:
        df = pd.read_csv(os.path.join(DATA_DIR, GT_MAP[cam]), header=None)
        all_ids.extend(df[1].unique())
    
    unique_ids = np.unique([id for id in all_ids if id >= 0])
    np.random.shuffle(unique_ids)
    split = int(len(unique_ids) * TRAIN_RATIO)
    train_ids = set(unique_ids[:split])
    test_ids = set(unique_ids[split:])
    
    print(f"Processing {len(unique_ids)} IDs. Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # 3.2 Cut Frames
    cam_id_map = {"cam1": 1, "cam2": 2, "cam3": 3}
    count_saved = 0
    
    for cam_name, v_file in VIDEO_MAP.items():
        v_path = os.path.join(DATA_DIR, v_file)
        gt_path = os.path.join(DATA_DIR, GT_MAP[cam_name])
        
        cap = cv2.VideoCapture(v_path)
        df = pd.read_csv(gt_path, header=None)
        groups = df.groupby(0)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % 2 != 0: continue # Skip every 2nd frame
            
            if frame_idx in groups.groups:
                rows = groups.get_group(frame_idx)
                for _, row in rows.iterrows():
                    tid = int(row[1])
                    if tid < 0: continue
                    
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                    x, y = max(0, x), max(0, y)
                    h_img, w_img = frame.shape[:2]
                    x2, y2 = min(w_img, x+w), min(h_img, y+h)
                    
                    crop = frame[y:y2, x:x2]
                    if crop.size == 0: continue
                    
                    fname = f"{tid:04d}_c{cam_id_map[cam_name]}s1_{frame_idx:06d}.jpg"
                    
                    if tid in train_ids:
                        cv2.imwrite(os.path.join(train_dir, fname), crop)
                    elif tid in test_ids:
                        if cam_name == "cam1":
                            cv2.imwrite(os.path.join(gallery_dir, fname), crop)
                        else:
                            cv2.imwrite(os.path.join(query_dir, fname), crop)
                    count_saved += 1
        cap.release()
    print(f"Extracted {count_saved} images.")

def restructure_for_market1501():
    """Step 4: Move folders to match TorchReID Market1501 structure"""
    # TorchReID ImageDataManager expects: root/market1501/Market-1501-v15.09.15/
    market_path = os.path.join(TEMP_ROOT, "market1501", "Market-1501-v15.09.15")
    os.makedirs(market_path, exist_ok=True)
    
    # Move folders
    for folder in ["bounding_box_train", "bounding_box_test", "query"]:
        src = os.path.join(TEMP_ROOT, folder)
        dst = os.path.join(market_path, folder)
        if os.path.exists(src):
            shutil.move(src, dst)
            
    print(f"Data restructured at {os.path.join(TEMP_ROOT, 'market1501')}")

def train_osnet_task(**kwargs):
    """Step 5: Run Training"""
    version = kwargs['ti'].xcom_pull(key='osnet_version', task_ids='get_version')
    dataset_root = os.path.join(TEMP_ROOT, "market1501")
    save_dir = os.path.join(MODELS_DIR, "osnet", f"osnet_{version}")
    
    print(f"Training Start. Version: {version}. Save to: {save_dir}")
    
    datamanager = data.ImageDataManager(
        root=dataset_root,
        sources="market1501",
        targets="market1501",
        height=256, width=128,
        batch_size_train=BATCH_SIZE,
        batch_size_test=32,
        transforms=["random_flip", "random_erase"]
    )
    
    model = models.build_model(
        name=BASE_MODEL_NAME,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=True
    )
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': model = model.cuda()
    
    optimizer = optim.build_optimizer(model, optim="adam", lr=LR)
    scheduler = optim.build_lr_scheduler(optimizer, lr_scheduler="single_step", stepsize=20)
    
    reid_engine = engine.ImageTripletEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )
    
    reid_engine.run(
        save_dir=save_dir,
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )
    return save_dir

def cleanup_temp():
    """Step 6: Cleanup"""
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
        print("Temp data cleaned.")

with DAG(
    'reid_retrain_detailed',
    default_args=default_args,
    description='Granular ReID Training Pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['reid', 'granular']
) as dag:

    t1_check = PythonOperator(
        task_id='check_resources',
        python_callable=check_resources
    )

    t2_ver = PythonOperator(
        task_id='get_version',
        python_callable=calculate_version
    )

    t3_extract = PythonOperator(
        task_id='extract_frames',
        python_callable=extract_frames_from_video
    )

    t4_struct = PythonOperator(
        task_id='organize_market1501',
        python_callable=restructure_for_market1501
    )

    t5_train = PythonOperator(
        task_id='train_model',
        python_callable=train_osnet_task
    )

    t6_clean = PythonOperator(
        task_id='cleanup_temp_data',
        python_callable=cleanup_temp,
        trigger_rule='all_done' # Chạy kể cả khi train fail để dọn rác
    )

    t1_check >> t2_ver >> t3_extract >> t4_struct >> t5_train >> t6_clean