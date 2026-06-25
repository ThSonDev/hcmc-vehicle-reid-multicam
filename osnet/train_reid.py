import os
import torch
from torchreid import data, models, optim, engine

print(">>> Connected to torchreid successfully!")

# --- 2. TUNING CONFIG ---

# Point directly at the market1501 folder inside the custom reid dataset
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reid_cam1_2_3")
MODEL_NAME = "osnet_x1_0"
MAX_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.0003
SAVE_DIR = "log_osnet_finetune_cam1_2_3_test"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Starting training on device: {device}")

    # 3. Init DataManager
    # Dropped the 'download' arg (it raises TypeError)
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

    # 4. Build Model
    print(f">>> Initializing backbone {MODEL_NAME}...")
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

    # 6. Init Engine
    reid_engine = engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # 7. Run Training
    reid_engine.run(
        save_dir=SAVE_DIR,
        max_epoch=MAX_EPOCHS,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

if __name__ == "__main__":
    main()