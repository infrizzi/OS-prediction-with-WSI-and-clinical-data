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

from src.datasets.wsi_dataset import VisualDataset
from src.models.wsi_abmil import ABMILRegressor

# Path setup
BASE_DIR = PROJECT_ROOT
TEST_DATA_DIR = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits\test")
JSON_PATH = BASE_DIR / "data" / "processed" / "patient_slide_association.json"
SCALER_PATH = BASE_DIR / "data" / "processed" / "target_scaler.pkl"

# Modular paths
ABMIL_PATH = BASE_DIR / "outputs" / "checkpoints" / "abmil_encoder.pth"
HEAD_PATH  = BASE_DIR / "outputs" / "checkpoints" / "wsi_head.pth"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load Scaler
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        return
    target_scaler = joblib.load(SCALER_PATH)

    # 2. Test dataset and loader
    test_ds = VisualDataset(TEST_DATA_DIR, JSON_PATH)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Model initialization and loading
    model = ABMILRegressor(input_dim=768, hidden_dim=256).to(device)
    
    # Modular weights loading
    if ABMIL_PATH.exists() and HEAD_PATH.exists():
        model.abmil.load_state_dict(torch.load(ABMIL_PATH, map_location=device))
        model.head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        print("Visual model weights (ABMIL + Head) loaded successfully.")
    else:
        print(f"ERROR: Checkpoints not found at {ABMIL_PATH} or {HEAD_PATH}")
        return
        
    model.eval()

    all_preds_std = []
    all_labels_std = []
    
    print(f"Starting inference on {len(test_ds)} samples...")

    # 4. Prediction loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            # ABMIL returns (prediction, attention_weights))
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
    c_index = concordance_index(labels_months, preds_months)

    # 7. Final report
    print("\n" + "="*45)
    print("      VISUAL UNIMODAL TEST REPORT ")
    print("="*45)
    print(f"MAE     : {mae:.2f} mesi")
    print(f"RMSE    : {rmse:.2f} mesi")
    print(f"R²      : {r2:.4f}")
    print(f"C-Index : {c_index:.4f}")
    print("-" * 45)
    
    # Esempi
    print("Sample Predictions:")
    for i in range(min(10, len(preds_months))): 
        diff = preds_months[i] - labels_months[i]
        print(f"Real: {labels_months[i]:6.1f} | Pred: {preds_months[i]:6.1f} | Err: {diff:6.1f}")
    print("="*45)

if __name__ == "__main__":
    evaluate()