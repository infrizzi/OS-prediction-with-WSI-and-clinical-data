import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Importiamo i pezzi dai vari moduli
from src.datasets.clinical_dataset import ClinicalDataset
from src.models.clinical_mlp import ClinicalMLP
from src.trainers.clinical_mlp_trainer import Trainer

# Path dinamici basati sulla tua struttura
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAVE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints" / "clinical_model_v1.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Datasets & Loaders
    train_ds = ClinicalDataset(DATA_DIR / "train_data.pt")
    val_ds   = ClinicalDataset(DATA_DIR / "val_data.pt")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    # 2. Modello, Loss e Optimizer
    model = ClinicalMLP(input_dim=463).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.L1Loss()

    # 3. Training Loop tramite Trainer
    trainer = Trainer(model, optimizer, criterion, device)
    
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(1, 101): 
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch:02d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            counter = 0
            # Salva solo il modello migliore
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping all'epoca {epoch}")
                break

if __name__ == "__main__":
    main()