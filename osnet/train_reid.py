import os
import torch
from torchreid import data, models, optim, engine

print(">>> Đã kết nối với torchreid thành công!")

# --- 2. TUNING CONFIG ---

# Trỏ trực tiếp vào thư mục market1501 bên trong reid_dataset_custom
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reid_cam1_2_3")
MODEL_NAME = "osnet_x1_0"
MAX_EPOCHS = 6
BATCH_SIZE = 32
LR = 0.0003
SAVE_DIR = "log_osnet_finetune_cam1_2_3"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Bắt đầu huấn luyện trên thiết bị: {device}")

    # 3. Khởi tạo DataManager
    # Bỏ tham số 'download' gây lỗi TypeError
    datamanager = data.ImageDataManager(
        root=DATASET_ROOT,
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=BATCH_SIZE,
        batch_size_test=32,
        num_instances=4,
        transforms=["random_flip", "random_erase", "color_jitter"]
    )

    # 4. Xây dựng Model
    print(f">>> Đang khởi tạo backbone {MODEL_NAME}...")
    model = models.build_model(
        name=MODEL_NAME,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=True
    )
    
    if device == 'cuda':
        model = model.cuda()

    # 5. Optimizer & Scheduler
    optimizer = optim.build_optimizer(model, optim="adam", lr=LR)
    scheduler = optim.build_lr_scheduler(
        optimizer, 
        lr_scheduler="single_step", 
        stepsize=20, 
        gamma=0.1
    )

    # 6. Khởi tạo Engine
    reid_engine = engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # 7. Chạy Huấn Luyện
    reid_engine.run(
        save_dir=SAVE_DIR,
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

if __name__ == "__main__":
    main()