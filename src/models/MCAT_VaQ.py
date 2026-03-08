import torch
import torch.nn as nn


class MCAT(nn.Module):
    """
    MCAT with configurable fusion strategy:
      - fusion="early": early fusion only (standard MCAT)
      - fusion="late": late fusion only (clinical head + visual head)
      - fusion="both": compute all and combine 

    Notes:
      - clinical_encoder: ClinicalEncoder (outputs [B, d_clin])
      - abmil: ABMIL aggregator (outputs [B, d_vis])
      - cross_attention: CrossAttention (clinical->visual)
      - clinical_head: ClinicalRegressionHead (d_clin -> 1)  
      - visual_head: RegressionHead (d_vis -> 1)             
    """
    def __init__(
        self,
        clinical_encoder,
        abmil_vis,
        abmil_clin,
        cross_attention,
        d_clin,
        d_vis,
        clinical_head=None,
        visual_head=None,
        fusion: str = "early",          # "early" | "late" | "both"
        late_strategy: str = "weighted", # "avg" | "weighted"
        dropout: float = 0.3,
    ):
        super().__init__()

        assert fusion in {"early", "late", "both"}
        assert late_strategy in {"avg", "weighted"}

        self.clinical_encoder = clinical_encoder
        self.cross_attention = cross_attention
        self.abmil_vis = abmil_vis
        self.abmil_clin = abmil_clin

        self.fusion = fusion
        self.late_strategy = late_strategy

        # Reuse unimodal heads 
        self.clinical_head = clinical_head  # ClinicalRegressionHead (d_clin->1)
        self.visual_head = visual_head      # RegressionHead (d_vis->1)

        # Early-fusion multimodal head
        self.regression_head = nn.Sequential(
            nn.Linear(d_clin + d_vis, 256), 
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),           
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)              
        )

        # Learnable weights for late fusion
        if self.late_strategy == "weighted":
            # weights for (clinical_pred, visual_pred, early_pred) in "both"
            # or (clinical_pred, visual_pred) in "late"
            n = 3 if fusion == "both" else 2
            initial_logits = torch.zeros(n)

            if n == 2:
            # Assuming: [0] Clinical, [1] Visual
            # softmax(0, 1) = [0.27, 0.73]
                initial_logits[0] = 1.0
                initial_logits[1] = 0.0 
        
            # From tensor to nn.Parameter so that they are learnable
            self._late_logits = nn.Parameter(initial_logits)

    def _compute_clinical_path(self, clin_emb, vis_x):
        """
        vis_x: [B, N, d_vis]
        """
        # 1. Co-Attention: clinical-guided visual contextualization
        # vis_guided: [B, N, d_vis]
        clin_guided, cross_attn_weights = self.cross_attention(clin_emb, vis_x)

        # 2. ABMIL: patch aggregation with attention-based MIL pooling
        # vis_emb: [B, d_vis], abmil_attn: [B, N]
        clin_new, abmil_attn = self.abmil_clin(clin_guided) 
        
        return clin_new, cross_attn_weights, abmil_attn

    def _late_fuse(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        preds: list of tensors each [B, 1]
        """
        if self.late_strategy == "avg":
            return torch.stack(preds, dim=0).mean(dim=0)

        # weighted: softmax over learnable logits
        w = torch.softmax(self._late_logits, dim=0)               # [k]
        stacked = torch.stack(preds, dim=0)                       # [k, B, 1]
        
        # weighted sum over k
        return (w.view(-1, 1, 1) * stacked).sum(dim=0)            # [B, 1]

    def forward(self, clin_x, vis_x):
        """
        clin_x: [B, D_in]
        vis_x:  [B, N, d_vis]
        """
        # 1) Clinical path
        clin_emb = self.clinical_encoder(clin_x)                  # [B, d_clin]

        # 2) Visual path (cross-att + ABMIL)
        clin_2_emb, attn, abmil_attn = self._compute_clinical_path(clin_emb, vis_x)  # [B,d_vis]
        vis_emb, _ = self.abmil_vis(vis_x)                          # [B, d_vis]

        out_info = {
            "cross_attention": attn,
            "abmil_attention": abmil_attn
        }

        # ---------- EARLY FUSION ----------
        early_pred = None
        if self.fusion in {"early", "both"}:
            fused = torch.cat([clin_emb, vis_emb], dim=1)         # [B, d_clin + d_vis]
            early_pred = self.regression_head(fused)              # [B, 1]
            out_info["early_pred"] = early_pred

        # ---------- LATE FUSION ----------
        late_pred = None
        if self.fusion in {"late", "both"}:
            if self.clinical_head is None or self.visual_head is None:
                raise ValueError(
                    "For fusion='late' or 'both' you must pass clinical_head and visual_head "
                    "(reuse unimodal regression heads)."
                )

            clin_pred = self.clinical_head(clin_2_emb)              # [B, 1]
            vis_pred = self.visual_head(vis_emb)                  # [B, 1]

            out_info["clinical_pred"] = clin_pred
            out_info["visual_pred"] = vis_pred

            if self.fusion == "late":
                late_pred = self._late_fuse([clin_pred, vis_pred])      # [B, 1]
            else:
                # both: fuse clinical + visual + early
                late_pred = self._late_fuse([clin_pred, vis_pred, early_pred])  # [B, 1]

            out_info["late_pred"] = late_pred

        # Return according to fusion mode
        if self.fusion == "early":
            return early_pred, out_info
        else:
            return late_pred, out_info