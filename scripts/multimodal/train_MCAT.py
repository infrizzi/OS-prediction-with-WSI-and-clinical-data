import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# =========================
# Add project root to PYTHONPATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ====== IMPORTS (adatta ai tuoi path reali) ======
from src.models.MCAT import MCAT                    # la classe MCAT che hai
from src.models.cross_attention import CrossAttention
from src.models.clinical_mlp import ClinicalEncoder, ClinicalRegressionHead
from src.models.wsi_abmil import ABMIL, RegressionHead
from src.trainers.MCAT_trainer import MCATTrainer

# Dataset multimodale: deve restituire (clin_x, vis_x, y)
from src.datasets.MCAT_dataset import MCATDataset   # <--- se non esiste ancora, lo crei (o adattalo)

# =========================
# CONFIG
# =========================
DATA_DIR = PROJECT_ROOT / "data" / "multimodal_splits"   # esempio
TRAIN_FILE = DATA_DIR / "train.pt"
VAL_FILE   = DATA_DIR / "val.pt"

CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "mcat"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Pesi unimodali (se vuoi inizializzare da quelli)
UNIMODAL_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CLIN_ENCODER_CKPT = UNIMODAL_DIR / "clinical_encoder.pth"
CLIN_HEAD_CKPT    = UNIMODAL_DIR / "clinical_head.pth"
ABMIL_CKPT        = UNIMODAL_DIR / "abmil_encoder.pth"
WSI_HEAD_CKPT     = UNIMODAL_DIR / "wsi_head.pth"

# Fusion
FUSION_MODE = "both"          # "early" | "late" | "both"
LATE_STRATEGY = "weighted"    # "avg" | "weighted"

# Train hyperparams
BATCH_SIZE = 1                # consigliato se N patch variabile
LR = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 200
PATIENCE = 15
ACCUM_STEPS = 8               # simile al tuo trainer visivo

# Freeze schedule (consigliato all’inizio)
FREEZE_CLIN_ENCODER = True
FREEZE_ABMIL = True


def save_mcat_checkpoints(model: MCAT, out_dir: Path):
    """
    Salva ogni componente in un file separato.
    """
    torch.save(model.clinical_encoder.state_dict(), out_dir / "clinical_encoder.pth")
    torch.save(model.abmil.state_dict(), out_dir / "abmil_encoder.pth")
    torch.save(model.cross_attention.state_dict(), out_dir / "cross_attention.pth")
    torch.save(model.regression_head.state_dict(), out_dir / "mcat_head.pth")

    # Se late/both, salva anche le head unimodali riusate (se presenti)
    if getattr(model, "clinical_head", None) is not None:
        torch.save(model.clinical_head.state_dict(), out_dir / "clinical_head.pth")
    if getattr(model, "visual_head", None) is not None:
        torch.save(model.visual_head.state_dict(), out_dir / "wsi_head.pth")

    # Se weighted late fusion, salva i logits
    if hasattr(model, "_late_logits"):
        torch.save(model._late_logits.detach().cpu(), out_dir / "late_logits.pt")


def maybe_load(module, path: Path):
    if path.exists():
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded: {path.name}")
    else:
        print(f"[WARN] Not found: {path}")


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # 1) Dataset & Loader
    # =========================
    train_ds = MCATDataset(TRAIN_FILE)
    val_ds   = MCATDataset(VAL_FILE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Infer dims from a sample
    clin_x0, vis_x0, y0 = train_ds[0]
    d_in = clin_x0.shape[-1]     # [D_in]
    d_vis = vis_x0.shape[-1]     # [N, D_vis]
    d_clin = 64                  # scelto da te (embedding_dim del ClinicalEncoder)
    d_model = 128                # cross-att internal dim

    # =========================
    # 2) Build modules
    # =========================
    clinical_encoder = ClinicalEncoder(input_dim=d_in, embedding_dim=d_clin)
    abmil = ABMIL(input_dim=d_vis, hidden_dim=256, dropout=0.5)
    cross_attention = CrossAttention(d_clin=d_clin, d_vis=d_vis, d_model=d_model)

    # Heads riusate per late fusion
    clinical_head = ClinicalRegressionHead(embedding_dim=d_clin) if FUSION_MODE in {"late", "both"} else None
    visual_head = RegressionHead(input_dim=d_vis, dropout=0.5) if FUSION_MODE in {"late", "both"} else None

    model = MCAT(
        clinical_encoder=clinical_encoder,
        abmil=abmil,
        cross_attention=cross_attention,
        d_clin=d_clin,
        d_vis=d_vis,
        clinical_head=clinical_head,
        visual_head=visual_head,
        fusion=FUSION_MODE,
        late_strategy=LATE_STRATEGY,
        dropout=0.3,
    ).to(device)

    # =========================
    # 3) Load unimodal weights (optional but recommended)
    # =========================
    maybe_load(model.clinical_encoder, CLIN_ENCODER_CKPT)
    maybe_load(model.abmil, ABMIL_CKPT)

    if FUSION_MODE in {"late", "both"}:
        maybe_load(model.clinical_head, CLIN_HEAD_CKPT)
        maybe_load(model.visual_head, WSI_HEAD_CKPT)

    # =========================
    # 4) Freeze encoders (recommended at start)
    # =========================
    if FREEZE_CLIN_ENCODER:
        set_requires_grad(model.clinical_encoder, False)

    if FREEZE_ABMIL:
        set_requires_grad(model.abmil, False)

    # =========================
    # 5) Optimizer: only trainable params
    # =========================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=1.0)

    # =========================
    # 6) Trainer
    # =========================
    trainer = MCATTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accumulation_steps=ACCUM_STEPS
    )

    # =========================
    # 7) Early stopping loop
    # =========================
    best_val = float("inf")
    counter = 0

    print(f"Starting MCAT training | fusion={FUSION_MODE} | late={LATE_STRATEGY}")
    for epoch in range(1, EPOCHS + 1):
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)

        print(f"Epoch {epoch:03d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            counter = 0
            save_mcat_checkpoints(model, CKPT_DIR)
            print("  --> Saved best MCAT checkpoints")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()