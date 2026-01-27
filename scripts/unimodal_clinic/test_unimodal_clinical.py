import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import custom modules 
from src.datasets.clinical_dataset import ClinicalDataset
from src.models.clinical_mlp import ClinicalMLP

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "splits" / "test_data.pt"
MODEL_PATH = BASE_DIR / "outputs" / "checkpoints" / "clinical_model_v1.pth"
SCALER_PATH = BASE_DIR / "data" / "processed" / "target_scaler.pkl"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento Scaler e Dataset
    if not SCALER_PATH.exists():
        print(f"ERRORE: Scaler del target non trovato in {SCALER_PATH}")
        return

    target_scaler = joblib.load(SCALER_PATH)
    test_ds = ClinicalDataset(DATA_PATH)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Inizializzazione Modello e caricamento pesi
    input_dim = test_ds[0][0].shape[0]
    model = ClinicalMLP(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds_std = []
    all_labels_std = []

    # 3. Fase di Inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            all_preds_std.extend(outputs.cpu().numpy().flatten())
            all_labels_std.extend(labels.cpu().numpy().flatten())

    # 4. DE-STANDARDIZZAZIONE
    # Convertiamo in array 2D per lo scaler [[val1], [val2], ...]
    all_preds_std = np.array(all_preds_std).reshape(-1, 1)
    all_labels_std = np.array(all_labels_std).reshape(-1, 1)

    # Trasformazione inversa per tornare ai mesi reali
    preds_months = target_scaler.inverse_transform(all_preds_std).flatten()
    labels_months = target_scaler.inverse_transform(all_labels_std).flatten()

    # 5. Calcolo delle Metriche su mesi REALI
    mae = mean_absolute_error(labels_months, preds_months)
    rmse = np.sqrt(mean_squared_error(labels_months, preds_months))
    r2 = r2_score(labels_months, preds_months)

    # 6. Report Finale
    print("\n" + "="*40)
    print("      REPORT VALUTAZIONE CLINICA")
    print("="*40)
    print(f"MAE (Errore Medio Assoluto):  {mae:.2f} mesi")
    print(f"RMSE (Radice Errore Quadr.): {rmse:.2f} mesi")
    print(f"R² (Bontà del Fit):           {r2:.4f}")
    print("-" * 40)
    print("Esempi Predizioni (Mesi Reali):")
    for i in range(10): # Vediamone 10 per sicurezza
        diff = preds_months[i] - labels_months[i]
        print(f"Reale: {labels_months[i]:5.1f} | Predetto: {preds_months[i]:5.1f} | Errore: {diff:5.1f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()