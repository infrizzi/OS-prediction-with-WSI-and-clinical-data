import torch
import numpy as np
import joblib
import sys
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index
from sklearn.utils import resample 

# =========================
# Add project root to PYTHONPATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import moduli MCAT
from src.models.MCAT import MCAT
from src.models.cross_attention import CrossAttention
from src.models.clinical_mlp import ClinicalEncoder, ClinicalRegressionHead
from src.models.wsi_abmil import ABMIL, RegressionHead
from src.datasets.MCAT_dataset import MCATDataset

# =========================
# PATH SETUP
# =========================
BASE_DIR = PROJECT_ROOT
TEST_CLINICAL_PATH = BASE_DIR / "data" / "splits" / "test_data.pt"
TEST_VISUAL_DIR = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits\test")
SCALER_PATH = BASE_DIR / "data" / "processed" / "target_scaler.pkl"

# Directory Checkpoints MCAT
MCAT_CKPT_DIR = BASE_DIR / "outputs" / "checkpoints" / "mcat_both_weighted"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MCAT on device: {device}")

    # 1. Caricamento Scaler
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        return
    target_scaler = joblib.load(SCALER_PATH)

    # 2. Dataset e Loader
    test_ds = MCATDataset(TEST_CLINICAL_PATH, TEST_VISUAL_DIR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Dataset Test pronto: {len(test_ds)} campioni.")

    # 3. Inizializzazione Modello
    clin_x0, vis_x0, _ = test_ds[0]
    d_in = clin_x0.shape[-1]
    d_vis = vis_x0.shape[-1]
    d_clin, d_model = 64, 128

    clinical_encoder = ClinicalEncoder(input_dim=d_in, embedding_dim=d_clin)
    abmil = ABMIL(input_dim=d_vis, hidden_dim=256)
    cross_attention = CrossAttention(d_clin=d_clin, d_vis=d_vis, d_model=d_model)
    
    clinical_head = ClinicalRegressionHead(embedding_dim=d_clin)
    visual_head = RegressionHead(input_dim=d_vis)

    model = MCAT(
        clinical_encoder=clinical_encoder,
        abmil=abmil,
        cross_attention=cross_attention,
        d_clin=d_clin,
        d_vis=d_vis,
        clinical_head=clinical_head,
        visual_head=visual_head,
        fusion="both",
        late_strategy="weighted"
    ).to(device)

    # 4. Caricamento Pesi
    try:
        model.clinical_encoder.load_state_dict(torch.load(MCAT_CKPT_DIR / "clinical_encoder.pth"))
        model.abmil.load_state_dict(torch.load(MCAT_CKPT_DIR / "abmil_encoder.pth"))
        model.cross_attention.load_state_dict(torch.load(MCAT_CKPT_DIR / "cross_attention.pth"))
        model.clinical_head.load_state_dict(torch.load(MCAT_CKPT_DIR / "clinical_head.pth"))
        model.visual_head.load_state_dict(torch.load(MCAT_CKPT_DIR / "wsi_head.pth"))
        model.regression_head.load_state_dict(torch.load(MCAT_CKPT_DIR / "mcat_head.pth"))
        
        if (MCAT_CKPT_DIR / "late_logits.pt").exists():
             model._late_logits = torch.nn.Parameter(torch.load(MCAT_CKPT_DIR / "late_logits.pt").to(device))
             
        print("MCAT modular weights loaded successfully.")
    except Exception as e:
        print(f"ERROR during weight loading: {e}")
        return

    model.eval()

    # Liste per raccogliere i risultati
    results = {
        "fused": {"preds": [], "labels": []},
        "clin":  {"preds": [], "labels": []},
        "vis":   {"preds": [], "labels": []}
    }

    # 5. Loop di Inferenza
    print("Inference in progress...")
    with torch.no_grad():
        for clin_x, vis_x, y in test_loader:
            clin_x = clin_x.to(device)
            vis_x = vis_x.to(device)
            pred_fused, out_info = model(clin_x, vis_x)

            results["fused"]["preds"].append(pred_fused.cpu().item())
            results["fused"]["labels"].append(y.item())
            results["clin"]["preds"].append(out_info["clinical_pred"].cpu().item())
            results["clin"]["labels"].append(y.item())
            results["vis"]["preds"].append(out_info["visual_pred"].cpu().item())
            results["vis"]["labels"].append(y.item())

    # 6. De-standardizzazione e Report con Bootstrap
    def get_metrics_with_bootstrap(y_true, y_pred, n_iterations=1000):
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)
        
        # Inverti scaling per avere mesi reali
        true_m = target_scaler.inverse_transform(y_true).flatten()
        pred_m = target_scaler.inverse_transform(y_pred).flatten()
        
        # Metriche "Point Estimate" (quelle originali)
        mae_point = mean_absolute_error(true_m, pred_m)
        r2_point = r2_score(true_m, pred_m)
        cindex_point = concordance_index(true_m, pred_m)

        # Bootstrap Loop
        boot_c = []
        boot_mae = []
        for _ in range(n_iterations):
            # Ricampionamento con ripetizione
            indices = resample(np.arange(len(true_m)), replace=True)
            if len(np.unique(true_m[indices])) < 2: continue # Salta campioni non validi
            
            boot_c.append(concordance_index(true_m[indices], pred_m[indices]))
            boot_mae.append(mean_absolute_error(true_m[indices], pred_m[indices]))

        return {
            "mae": mae_point,
            "mae_std": np.std(boot_mae),
            "r2": r2_point,
            "cindex": cindex_point,
            "cindex_std": np.std(boot_c),
            "raw_preds": pred_m,
            "raw_labels": true_m
        }

    print(f"Calculating metrics and bootstrap SD (1000 iterations)...")
    m_fused = get_metrics_with_bootstrap(results["fused"]["labels"], results["fused"]["preds"])
    m_clin = get_metrics_with_bootstrap(results["clin"]["labels"], results["clin"]["preds"])
    m_vis = get_metrics_with_bootstrap(results["vis"]["labels"], results["vis"]["preds"])

    # 7. Final Report
    print("\n" + "="*75)
    print(f"{'MODALITY':<15} | {'C-INDEX (±SD)':<18} | {'MAE (±SD) Months':<18} | {'R2':<8}")
    print("-" * 75)
    print(f"{'Clinical Only':<15} | {m_clin['cindex']:.4f} ± {m_clin['cindex_std']:.4f} | {m_clin['mae']:.2f} ± {m_clin['mae_std']:.2f}     | {m_clin['r2']:.4f}")
    print(f"{'Visual Only':<15} | {m_vis['cindex']:.4f} ± {m_vis['cindex_std']:.4f} | {m_vis['mae']:.2f} ± {m_vis['mae_std']:.2f}     | {m_vis['r2']:.4f}")
    print(f"{'MCAT Fused':<15} | {m_fused['cindex']:.4f} ± {m_fused['cindex_std']:.4f} | {m_fused['mae']:.2f} ± {m_fused['mae_std']:.2f}     | {m_fused['r2']:.4f}")
    print("="*75)

    # Analisi pesi finali
    with torch.no_grad():
        w = torch.softmax(model._late_logits, dim=0).cpu().numpy()
        print(f"Final Weights -> CLIN: {w[0]*100:.1f}% | VISUAL: {w[1]*100:.1f}%")

    # Esempi (Fusi)
    print("\nSample Predictions (MCAT Fused):")
    for i in range(min(10, len(m_fused['raw_preds']))):
        diff = m_fused['raw_preds'][i] - m_fused['raw_labels'][i]
        print(f"Real: {m_fused['raw_labels'][i]:6.1f} | Pred: {m_fused['raw_preds'][i]:6.1f} months | Diff: {diff:.2f}")

if __name__ == "__main__":
    evaluate()