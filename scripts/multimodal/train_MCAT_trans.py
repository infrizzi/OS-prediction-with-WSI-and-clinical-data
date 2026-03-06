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

# Import MCAT modules
from src.models.MCAT import MCAT                   
from src.models.cross_attention_VaQ import CrossAttention
from src.models.clinical_mlp import ClinicalEncoder, ClinicalRegressionHead
from src.models.wsi_transmil import TransMIL, RegressionHead
from src.trainers.MCAT_trainer import MCATTrainer
from src.datasets.MCAT_dataset import MCATDataset  

# =========================
# PATH SETUP
# =========================
TRAIN_PATH_CLINICAL = PROJECT_ROOT / "data" / "splits" / "train_data.pt"
VAL_PATH_CLINICAL  = PROJECT_ROOT / "data" / "splits" / "val_data.pt"
TRAIN_VISUAL_DIR = Path(r"/homes/lpaladino/visual_splits/train")
VAL_VISUAL_DIR  = Path(r"/homes/lpaladino/visual_splits/val")

# MCAT checkpoint directory
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "transmil" / "mcat_VaQ_noconcat_full"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Unimodal checkpoints
UNIMODAL_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CLIN_ENCODER_CKPT = UNIMODAL_DIR / "clinical_encoder.pth"
CLIN_HEAD_CKPT    = UNIMODAL_DIR / "clinical_head.pth"
ABMIL_CKPT        = UNIMODAL_DIR / "transmil_encoder.pth"
WSI_HEAD_CKPT     = UNIMODAL_DIR / "wsi_transmil_head.pth"

# Fusion
FUSION_MODE = "late"          # "early" | "late" | "both"
LATE_STRATEGY = "weighted"    # "avg" | "weighted"

# Train hyperparams
BATCH_SIZE = 1                
LR = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 200
DROPOUT = 0.2
PATIENCE = 10
ACCUM_STEPS = 128              

# Warmup first epochs
WARMUP_MODE = False           # If True, ABMIL and clinical encoder are frozen for the firsts epochs -> only cross-attention and heads are trained, then all is unfrozen and trained together
WARMUP_EPOCHS = 10 

# ======================================================================================================================================================

# Save new checkpoints for MCAT model 
# (clinical encoder, abmil, cross-attention, fusion head, and optionally unimodal heads and late fusion logits)
def save_mcat_checkpoints(model: MCAT, out_dir: Path):

    torch.save(model.clinical_encoder.state_dict(), out_dir / "clinical_encoder.pth")
    torch.save(model.abmil.state_dict(), out_dir / "transmil_encoder.pth")
    torch.save(model.cross_attention.state_dict(), out_dir / "cross_attention.pth")
    torch.save(model.regression_head.state_dict(), out_dir / "mcat_head.pth")

    # If late/both, also save unimodal heads
    if getattr(model, "clinical_head", None) is not None:
        torch.save(model.clinical_head.state_dict(), out_dir / "clinical_head.pth")
    if getattr(model, "visual_head", None) is not None:
        torch.save(model.visual_head.state_dict(), out_dir / "wsi_transmil_head.pth")

    # if weighted, also save fusion logits
    if hasattr(model, "_late_logits"):
        torch.save(model._late_logits.detach().cpu(), out_dir / "late_logits.pt")


# Load module with the corresponding checkpoint, if it exists
def load_tensor(module, path: Path):
    if path.exists():
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded: {path.name}")
    else:
        print(f"[WARN] Not found: {path}")


# Utility function to freeze/unfreeze modules
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# Utility to print current fusion weights 
def print_fusion_weights(model):
    if not hasattr(model, "_late_logits"):
        print("Model doesn't use weighted late fusion strategy")
        return

    # Extract weights using softmax
    with torch.no_grad():
        weights = torch.softmax(model._late_logits, dim=0).cpu().numpy()
    
    if model.fusion == "late":
        # Order: [clinical_pred, visual_pred]
        print(f"    CLINICAL: {weights[0]*100:.2f}% | VISUAL:  {weights[1]*100:.2f}%")
    
    elif model.fusion == "both":
        # Order: [clinical_pred, visual_pred, early_pred]
        print(f"    CLINICAL: {weights[0]*100:.2f}% | VISUAL:  {weights[1]*100:.2f}% | EARLY:  {weights[2]*100:.2f}%")
    
    print("_"*30 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # 1) Dataset & Loader
    # =========================
    train_ds = MCATDataset(TRAIN_PATH_CLINICAL, TRAIN_VISUAL_DIR)
    val_ds   = MCATDataset(VAL_PATH_CLINICAL, VAL_VISUAL_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Infer dims from a sample
    clin_x0, vis_x0, y0 = train_ds[0]
    d_in = clin_x0.shape[-1]     # [D_in]
    d_vis = vis_x0.shape[-1]     # [N, D_vis]
    d_clin = 463                  # scelto da te (embedding_dim del ClinicalEncoder)
    d_model = 768                # cross-att internal dim
    n_head = 4                   # cross-att number of heads
    dropout_cross = 0.2          # cross-att dropout

    # =========================
    # 2) Build modules
    # =========================
    # MCAT core modules
    clinical_encoder = ClinicalEncoder(input_dim=d_in, embedding_dim=d_clin)
    abmil = TransMIL(input_dim=d_vis, hidden_dim=768, dropout=0.3)
    cross_attention = CrossAttention(d_clin=d_clin, d_vis=d_vis, d_model=d_model, n_heads=n_head, dropout=dropout_cross)

    # Unimodal heads only if late/both fusion
    clinical_head = ClinicalRegressionHead(embedding_dim=d_clin) if FUSION_MODE in {"late", "both"} else None
    visual_head = RegressionHead(input_dim=768, dropout=0.3) if FUSION_MODE in {"late", "both"} else None

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
        dropout=DROPOUT,
    ).to(device)

    # =========================
    # 3) Load unimodal weights
    # =========================
    load_tensor(model.clinical_encoder, CLIN_ENCODER_CKPT)
    load_tensor(model.abmil, ABMIL_CKPT)

    if FUSION_MODE in {"late", "both"}:
        load_tensor(model.clinical_head, CLIN_HEAD_CKPT)
        load_tensor(model.visual_head, WSI_HEAD_CKPT)

    # =========================
    # 4) Warmup first epochs
    # =========================
    if WARMUP_MODE:
        print(f"\n>>> PHASE 1: WARM-UP ({WARMUP_EPOCHS} epochs)")

    set_requires_grad(model.clinical_encoder, True)
    set_requires_grad(model.abmil, True)
    set_requires_grad(model.cross_attention, True)
    set_requires_grad(model.clinical_head, True)
    set_requires_grad(model.visual_head, True)
    

    # =========================
    # 5) Optimizer: only trainable params
    # =========================
    fusion_params = []
    head_params = []
    encoder_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Discriminate between late fusion logits and normal params
        if "_late_logits" in name:
            fusion_params.append(param)
        elif "clinical_head" in name or "visual_head" in name:
            head_params.append(param)
        elif "clinical_encoder" in name or "abmil" in name:
            encoder_params.append(param)
        else:
            base_params.append(param)

    # Define each learning rate
    param_groups = [
        {'params': base_params, 'lr': LR},           # standard LR (1e-4)
        {'params': fusion_params, 'lr': 1e-2},       # higher LR -> learning faster
        {'params': head_params, 'lr': LR},           # standard LR (1e-4)        
        {'params': encoder_params, 'lr': 1e-5},      # lower LR for encoders        
    ]

    optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
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
    is_full_training = False

    print(f"Starting MCAT training | fusion={FUSION_MODE} | late={LATE_STRATEGY}")
    for epoch in range(1, EPOCHS + 1):

        if epoch > WARMUP_EPOCHS and not is_full_training and WARMUP_MODE:
            print(f"\n>>> PHASE 2: FINE-TUNING ")
            set_requires_grad(model.clinical_encoder, True)
            set_requires_grad(model.abmil, True)
            
            # RE-INITIALIZE OPTIMIZER with different LR
            optimizer = torch.optim.Adam([
                # Encoders: LR very low because already well-trained
                {'params': model.clinical_encoder.parameters(), 'lr': 1e-5}, 
                {'params': model.abmil.parameters(), 'lr': 1e-5},

                # Cross-Attention e Head: standard LR
                {'params': model.cross_attention.parameters(), 'lr': LR},
                {'params': model.regression_head.parameters(), 'lr': LR},

                # Late fusion weights
                {'params': [model._late_logits], 'lr': 1e-2},
            ], weight_decay=5e-2) # Higher WD
            
            trainer.optimizer = optimizer
            is_full_training = True

        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)

        print(f"Epoch {epoch:03d} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            counter = 0
            save_mcat_checkpoints(model, CKPT_DIR)
            print("--> Saved best MCAT checkpoints")
            print_fusion_weights(model)
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()