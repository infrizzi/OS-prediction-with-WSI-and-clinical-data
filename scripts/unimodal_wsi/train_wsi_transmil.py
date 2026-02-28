import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# =========================
# Add project root to PYTHONPATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.datasets.wsi_dataset import VisualDataset
from src.models.wsi_transmil import TransMILRegressor
from src.trainers.wsi_trainer import VisualTrainer

# Path setup
JSON_PATH = PROJECT_ROOT / "data" / "processed" / "patient_slide_association.json"
TRAIN_DIR = Path(r"/homes/lpaladino/visual_splits/train")
VAL_DIR   = Path(r"/homes/lpaladino/visual_splits/val")
SAVE_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "visual_model.pth"

DROPOUT = 0.2
ACC_STEPS = 128

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Datasets & Loaders
    train_ds = VisualDataset(TRAIN_DIR, JSON_PATH)
    val_ds   = VisualDataset(VAL_DIR, JSON_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True) # Batch size 1 with variable patch count N
    val_loader   = DataLoader(val_ds, batch_size=1)

    # 2. Model, Loss and Optimizer
    model = TransMILRegressor(input_dim=train_ds[0][0].shape[1], hidden_dim=512, dropout=DROPOUT).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.transmil.parameters(), "lr": 1e-4},
            {"params": model.head.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3
    )

    criterion = nn.SmoothL1Loss()

    # 3. Training loop with Trainer
    trainer = VisualTrainer(model, optimizer, criterion, device, accumulation_steps=ACC_STEPS)
    
    # 4. Early Stopping Setup
    best_val_loss = float('inf')
    patience = 10 
    counter = 0
  
    # 5. Training Process
    print("Starting Visual Unimodal Training...")
    for epoch in range(1, 201): 
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch:02d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            counter = 0
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.transmil.state_dict(),
                    SAVE_PATH.with_name("transmil_encoder.pth"))

            torch.save(model.head.state_dict(),
                    SAVE_PATH.with_name("wsi_transmil_head.pth"))
            
            print(f"  --> Model saved")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping fired at epoch {epoch}")
                break
    
if __name__ == "__main__":
    main()