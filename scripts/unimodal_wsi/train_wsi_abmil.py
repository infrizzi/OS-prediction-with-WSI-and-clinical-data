import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Aggiunta del root del progetto al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.datasets.wsi_dataset import VisualDataset
from src.models.wsi_abmil import ABMILRegressor
from src.trainers.wsi_trainer import VisualTrainer

# Configurazione Percorsi
JSON_PATH = PROJECT_ROOT / "data" / "processed" / "patient_slide_association.json"
# Percorsi esterni per i tensori aggregati
TRAIN_DIR = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits\train")
VAL_DIR   = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits\val")
SAVE_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "visual_model_v1.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Datasets & Loaders (batch_size=1 è obbligatorio per N patch variabili)
    train_ds = VisualDataset(TRAIN_DIR, JSON_PATH)
    val_ds   = VisualDataset(VAL_DIR, JSON_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)

    # 2. Model, Loss and Optimizer
    model = ABMILRegressor(input_dim=train_ds[0][0].shape[1], hidden_dim=256).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()

    # 3. Training loop with Trainer
    trainer = VisualTrainer(model, optimizer, criterion, device, accumulation_steps=8)
    
    # 4. Early Stopping Setup
    best_val_loss = float('inf')
    patience = 15 
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
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> Model saved")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping fired at epoch {epoch}")
                break
    
if __name__ == "__main__":
    main()