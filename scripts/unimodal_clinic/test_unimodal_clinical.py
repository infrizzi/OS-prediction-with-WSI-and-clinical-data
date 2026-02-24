import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index
from sklearn.utils import resample # Per il calcolo della varianza statistica
import sys

# =========================
# Add project root to PYTHONPATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules 
from src.datasets.clinical_dataset import ClinicalDataset
from src.models.clinical_mlp import ClinicalMLP

# Path setup
BASE_DIR = PROJECT_ROOT
DATA_PATH = BASE_DIR / "data" / "splits" / "test_data_new.pt"

# Modular checkpoint paths
ENCODER_PATH = BASE_DIR / "outputs" / "checkpoints" / "clinical_encoder_new.pth"
HEAD_PATH = BASE_DIR / "outputs" / "checkpoints" / "clinical_head_new.pth"

# Scaler path for de-standardization
SCALER_PATH = BASE_DIR / "data" / "processed" / "target_scaler.pkl"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Clinical MLP on device: {device}")
    
    # 1. Test dataset and loader
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        return

    target_scaler = joblib.load(SCALER_PATH)
    test_ds = ClinicalDataset(DATA_PATH)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"Dataset Test pronto: {len(test_ds)} campioni.")

    # 2. Model initialization and loading
    input_dim = test_ds[0][0].shape[0]
    model = ClinicalMLP(input_dim=input_dim).to(device)
    
    # Modular weights loading
    if ENCODER_PATH.exists() and HEAD_PATH.exists():
        model.encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
        model.head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        print("Model weights loaded successfully from separate files.")
    else:
        print(f"ERROR: Checkpoints not found at {ENCODER_PATH} or {HEAD_PATH}")
        return

    model.eval()

    all_preds_std = []
    all_labels_std = []

    # 3. Prediction loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            all_preds_std.extend(outputs.cpu().numpy().flatten())
            all_labels_std.extend(labels.cpu().numpy().flatten())

    # 4. De-standardization
    all_preds_std = np.array(all_preds_std).reshape(-1, 1)
    all_labels_std = np.array(all_labels_std).reshape(-1, 1)

    preds_months = target_scaler.inverse_transform(all_preds_std).flatten()
    labels_months = target_scaler.inverse_transform(all_labels_std).flatten()

    # ==========================================================
    # 5. Bootstrap Calculation (1000 iterations)
    # ==========================================================
    
    n_iterations = 1000
    boot_cindex = []
    boot_mae = []

    print(f"Running Bootstrap (n={n_iterations})...")
    for _ in range(n_iterations):
        # Campionamento casuale con reinserimento
        indices = resample(np.arange(len(labels_months)), replace=True)
        
        # Check per evitare campioni con varianza nulla nel target
        if len(np.unique(labels_months[indices])) < 2:
            continue

        c = concordance_index(labels_months[indices], preds_months[indices])
        m = mean_absolute_error(labels_months[indices], preds_months[indices])
        
        boot_cindex.append(c)
        boot_mae.append(m)

    # Statistiche finali SD
    c_mean, c_std = np.mean(boot_cindex), np.std(boot_cindex)
    mae_mean, mae_std = np.mean(boot_mae), np.std(boot_mae)

    # Altre metriche puntuali
    rmse = np.sqrt(mean_squared_error(labels_months, preds_months))
    r2 = r2_score(labels_months, preds_months)

    # ==========================================================
    # 6. Final report
    # ==========================================================
    print("\n" + "="*65)
    print("      CLINICAL UNIMODAL TEST REPORT (WITH BOOTSTRAP) ")
    print("="*65)
    print(f"{'METRIC':<15} | {'VALUE ± SD':<20}")
    print("-" * 65)
    print(f"{'C-Index':<15} | {c_mean:.4f} ± {c_std:.4f}")
    print(f"{'MAE (Months)':<15} | {mae_mean:.2f} ± {mae_std:.2f}")
    print(f"{'RMSE':<15} | {rmse:.2f} mesi")
    print(f"{'R²':<15} | {r2:.4f}")
    print("-" * 65)
    
    print("Examples:")
    for i in range(min(5, len(preds_months))): 
        diff = preds_months[i] - labels_months[i]
        print(f"Real: {labels_months[i]:6.1f} | Pred: {preds_months[i]:6.1f} | Err: {diff:6.1f}")
    print("="*65)

if __name__ == "__main__":
    evaluate()