import os
import shutil
import sys
import numpy as np

# --- VÁ LỖI NUMPY > 1.20 ---
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
# ---------------------------

# --- CONFIG ---
GT_SOURCE_DIR = "./"           # Nơi chứa gt_cam1.txt
RES_SOURCE_DIR = "./results"   # Nơi chứa res_cam1.txt (Output của file trên)
TEMP_EVAL_ROOT = "./temp_eval_data" 

TRACKER_NAME = "MyLocalAlgo"
SEQS = {
    "cam1": {"w": 1920, "h": 1080},
    "cam2": {"w": 1920, "h": 1080},
    "cam3": {"w": 1440, "h": 1080}
}

def create_seqinfo(path, seq_name, w, h):
    content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=30
seqLength=10000
imWidth={w}
imHeight={h}
imExt=.jpg
"""
    with open(path, 'w') as f: f.write(content)

def setup_environment():
    print(f">>> Setup thư mục đánh giá tại: {TEMP_EVAL_ROOT}")
    if os.path.exists(TEMP_EVAL_ROOT): shutil.rmtree(TEMP_EVAL_ROOT)
    
    gt_root = os.path.join(TEMP_EVAL_ROOT, "gt") 
    tracker_root = os.path.join(TEMP_EVAL_ROOT, "trackers")
    
    # Cấu trúc folder tracker phẳng (do SKIP_SPLIT_FOL=True)
    tracker_data_dir = os.path.join(tracker_root, TRACKER_NAME, "data")
    os.makedirs(tracker_data_dir, exist_ok=True)

    valid_seqs = []
    for seq, info in SEQS.items():
        # Setup GT: gt/cam1/gt/gt.txt và gt/cam1/seqinfo.ini
        src_gt = os.path.join(GT_SOURCE_DIR, f"gt_{seq}.txt")
        if not os.path.exists(src_gt):
            print(f"⚠️ Thiếu GT: {src_gt} -> Skip {seq}")
            continue
            
        seq_dir = os.path.join(gt_root, seq)
        os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
        shutil.copy(src_gt, os.path.join(seq_dir, "gt", "gt.txt"))
        create_seqinfo(os.path.join(seq_dir, "seqinfo.ini"), seq, info['w'], info['h'])

        # Setup Tracker Result
        src_res = os.path.join(RES_SOURCE_DIR, f"res_{seq}.txt")
        dst_res = os.path.join(tracker_data_dir, f"{seq}.txt")
        if not os.path.exists(src_res):
            print(f"⚠️ Thiếu Result: {src_res} -> Tạo file rỗng")
            open(dst_res, 'w').close()
        else:
            shutil.copy(src_res, dst_res)
            
        valid_seqs.append(seq)
    return valid_seqs

def run_evaluation():
    try:
        import trackeval
    except ImportError:
        print("❌ Lỗi: Chưa cài thư viện 'trackeval'.")
        print("👉 Chạy: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
        return

    valid_seqs = setup_environment()
    if not valid_seqs: return

    print(f"\n>>> ĐANG TÍNH TOÁN METRICS CHO: {valid_seqs}")

    # Config TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['PRINT_CONFIG'] = False
    eval_config['TIME_PROGRESS'] = False
    eval_config['USE_PARALLEL'] = False # Tắt parallel để tránh lỗi trên một số máy

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = os.path.join(TEMP_EVAL_ROOT, 'gt')
    dataset_config['TRACKERS_FOLDER'] = os.path.join(TEMP_EVAL_ROOT, 'trackers')
    dataset_config['BENCHMARK'] = 'MotChallenge2DBox'
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['TRACKERS_TO_EVAL'] = [TRACKER_NAME]
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian'] 
    dataset_config['SEQ_INFO'] = {seq: None for seq in valid_seqs}
    dataset_config['SKIP_SPLIT_FOL'] = True # QUAN TRỌNG

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    # Chỉ định các metrics cần tính
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        metrics_list.append(metric())

    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == "__main__":
    run_evaluation()