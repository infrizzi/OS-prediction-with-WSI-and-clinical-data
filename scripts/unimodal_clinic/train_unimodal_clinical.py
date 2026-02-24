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

# Import custom modules 
from src.datasets.clinical_dataset import ClinicalDataset
from src.models.clinical_mlp import ClinicalMLP
from src.trainers.clinical_trainer import Trainer
from utils.Cox import CoxPHLoss

DATA_DIR = PROJECT_ROOT / "data" / "splits"
SAVE_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "clinical_model_v1.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Datasets & Loaders
    train_ds = ClinicalDataset(DATA_DIR / "train_data_new.pt")
    val_ds   = ClinicalDataset(DATA_DIR / "val_data_new.pt")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    # 2. Model, Loss and Optimizer
    model = ClinicalMLP(input_dim=train_ds[0][0].shape[0]).to(device)
    
    optimizer = torch.optim.Adam(
    [
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-2
)
    # criterion = nn.L1Loss()                  # Mean Absolute Error
    criterion = nn.SmoothL1Loss(beta=1.0)    # Huber loss
    # criterion = CoxPHLoss()                  # Cox Proportional Hazards Loss for survival analysis


    # 3. Training loop with Trainer
    trainer = Trainer(model, optimizer, criterion, device)
    
    # 4. Early Stopping Setup
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    # 5. Training Process
    for epoch in range(1, 101): 
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch:02d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            counter = 0
            # Save only the best model
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                model.encoder.state_dict(),
                SAVE_PATH.with_name("clinical_encoder_new.pth")
            )

            torch.save(
                model.head.state_dict(),
                SAVE_PATH.with_name("clinical_head_new.pth")
            )
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping fired at epoch {epoch}")
                break

if __name__ == "__main__":
    main()