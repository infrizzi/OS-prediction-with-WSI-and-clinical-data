import torch
import numpy as np
import joblib
import sys
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index

# =========================
# Add project root to PYTHONPATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules 
from src.datasets.wsi_dataset import VisualDataset
from src.models.wsi_abmil import ABMILRegressor

# Path setup
BASE_DIR = PROJECT_ROOT
# Percorso locale indicato per i file .pt di test
TEST_DATA_DIR = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits\test")
JSON_PATH = BASE_DIR / "data" / "processed" / "patient_slide_association.json"
MODEL_PATH = BASE_DIR / "outputs" / "checkpoints" / "visual_model_v1.pth"
SCALER_PATH = BASE_DIR / "data" / "processed" / "target_scaler.pkl"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load Scaler
    if not SCALER_PATH.exists():
        print(f"ERRORE: Scaler del target non trovato in {SCALER_PATH}")
        return
    target_scaler = joblib.load(SCALER_PATH)

    # 2. Test dataset and loader (Batch size 1 obbligatorio)
    test_ds = VisualDataset(TEST_DATA_DIR, JSON_PATH)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Model initialization and loading
    # Assicurati che input_dim coincida con CONCH (768)
    model = ABMILRegressor(input_dim=768, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds_std = []
    all_labels_std = []
    
    print(f"Starting inference on {len(test_ds)} samples...")

    # 4. Prediction loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            # L'ABMIL sputa (predizione, pesi_attenzione)
            outputs, _ = model(inputs.to(device))
            
            all_preds_std.append(outputs.cpu().item())
            all_labels_std.append(labels.cpu().item())

    # 5. Conversion and De-standardization
    all_preds_std = np.array(all_preds_std).reshape(-1, 1)
    all_labels_std = np.array(all_labels_std).reshape(-1, 1)

    preds_months = target_scaler.inverse_transform(all_preds_std).flatten()
    labels_months = target_scaler.inverse_transform(all_labels_std).flatten()

    # 6. Metrics Calculation
    mae = mean_absolute_error(labels_months, preds_months)
    rmse = np.sqrt(mean_squared_error(labels_months, preds_months))
    r2 = r2_score(labels_months, preds_months)
    
    # C-Index: richiede le predizioni (o rischi) e i tempi reali
    # Per lifelines, usiamo il negativo delle predizioni se le predizioni sono 'tempi di vita'
    # perché un valore alto di predizione = basso rischio
    c_index = concordance_index(labels_months, preds_months)

    # 7. Final report
    print("\n" + "="*45)
    print("           VISUAL UNIMODAL TEST REPORT")
    print("="*45)
    print(f"MAE  : {mae:.2f} mesi")
    print(f"RMSE : {rmse:.2f} mesi")
    print(f"R²   : {r2:.4f}")
    print(f"C-Index : {c_index:.4f}  <-- Metrica chiave")
    print("-" * 45)
    
    # Esempi
    print("Top 10 Predictions:")
    for i in range(10): 
        diff = preds_months[i] - labels_months[i]
        print(f"Real: {labels_months[i]:6.1f} | Pred: {preds_months[i]:6.1f} | Error: {diff:6.1f}")
    print("="*45)

if __name__ == "__main__":
    evaluate()