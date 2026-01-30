import cv2
import os
import pandas as pd
import numpy as np
import shutil

# --- CẤU HÌNH DỮ LIỆU ---

VIDEO_PATHS = {
    "cam1": "../data/cam1_half.mp4",
    "cam2": "../data/cam2_half.mp4",
    "cam3": "../data/cam3_half.mp4" 
}

ANNOTATION_PATHS = {
    "cam1": "../gt_cam1.txt",
    "cam2": "../gt_cam2.txt",
    "cam3": "../gt_cam3.txt"
}

OUTPUT_DIR = "reid_cam1_2_3" 
TRAIN_RATIO = 0.8  

def ensure_dir(path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

def main():
    # 1. Tạo cấu trúc thư mục
    train_dir = os.path.join(OUTPUT_DIR, "bounding_box_train")
    query_dir = os.path.join(OUTPUT_DIR, "query")          
    gallery_dir = os.path.join(OUTPUT_DIR, "bounding_box_test") 
    
    ensure_dir(train_dir)
    ensure_dir(query_dir)
    ensure_dir(gallery_dir)
    
    # 2. LẤY TẤT CẢ ID TỪ CẢ 3 CAMERAS (Sửa lỗi bỏ sót ID)
    all_ids_list = []
    for cam_name in ANNOTATION_PATHS:
        df_tmp = pd.read_csv(ANNOTATION_PATHS[cam_name], header=None)
        all_ids_list.extend(df_tmp[1].unique())
    
    all_ids = np.unique(all_ids_list)
    all_ids = all_ids[all_ids >= 0] # Lọc ID rác
    
    np.random.shuffle(all_ids)
    split_idx = int(len(all_ids) * TRAIN_RATIO)
    train_ids = set(all_ids[:split_idx])
    test_ids = set(all_ids[split_idx:])
    
    print(f"Tổng số ID từ 3 Cam: {len(all_ids)}. Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # 3. Xử lý video và lưu ảnh
    cam_map = {"cam1": 1, "cam2": 2, "cam3": 3} 
    
    for cam_name, v_path in VIDEO_PATHS.items():
        anno_path = ANNOTATION_PATHS[cam_name]
        print(f"Đang cắt ảnh {cam_name}...")
        
        cap = cv2.VideoCapture(v_path)
        df = pd.read_csv(anno_path, header=None)
        groups = df.groupby(0)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            if frame_idx % 2 != 0: continue # Giảm FPS để tránh trùng lặp quá nhiều

            if frame_idx in groups.groups:
                rows = groups.get_group(frame_idx)
                for _, row in rows.iterrows():
                    tid = int(row[1])
                    if tid < 0: continue
                    
                    x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                    x, y = max(0, x), max(0, y)
                    img_h, img_w = frame.shape[:2]
                    x2, y2 = min(img_w, x+w), min(img_h, y+h)
                    
                    crop = frame[y:y2, x:x2]
                    if crop.size == 0: continue
                    
                    fname = f"{tid:04d}_c{cam_map[cam_name]}s1_{frame_idx:06d}.jpg"
                    
                    if tid in train_ids:
                        cv2.imwrite(os.path.join(train_dir, fname), crop)
                    elif tid in test_ids:
                        # Chiến lược phân bổ cho tập Test:
                        # - Gallery: Dùng Cam 1 (Kho ảnh gốc)
                        # - Query: Dùng Cam 2 và Cam 3 (Ảnh thực tế đi qua để truy vấn)
                        if cam_name == "cam1":
                            cv2.imwrite(os.path.join(gallery_dir, fname), crop)
                        else:
                            cv2.imwrite(os.path.join(query_dir, fname), crop)
                            
        cap.release()

    print("Chuẩn bị dữ liệu 3 Cam hoàn tất!")

if __name__ == "__main__":
    main()

'''
    mkdir -p market1501/Market-1501-v15.09.15
    mv bounding_box_train market1501/Market-1501-v15.09.15/
    mv bounding_box_test market1501/Market-1501-v15.09.15/
    mv query market1501/Market-1501-v15.09.15/
    '''