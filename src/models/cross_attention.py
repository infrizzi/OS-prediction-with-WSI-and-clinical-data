import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model):
        super().__init__()
        # Query: Proietta la clinica
        self.q_proj = nn.Linear(d_clin, d_model)
        # Key: Proietta il visivo
        self.k_proj = nn.Linear(d_vis, d_model)
        # Value: Proietta il visivo (mantenendo d_vis per il residuo)
        self.v_proj = nn.Linear(d_vis, d_vis) 
        
        self.attn_dropout = nn.Dropout(0.3)

        # LayerNorm per stabilizzare il training post-somma
        self.norm = nn.LayerNorm(d_vis)
        
        self.scale = d_model ** -0.5

    def forward(self, clin_emb, vis_x):
        """
        clin_emb: [B, d_clin]
        vis_x:    [B, N, d_vis]
        """
        # --- 1. Preparazione Q, K, V ---
        # Q: [B, 1, d_model]
        Q = self.q_proj(clin_emb).unsqueeze(1)
        # K: [B, N, d_model]
        K = self.k_proj(vis_x)
        # V: [B, N, d_vis]
        V = self.v_proj(vis_x)
        
        # --- 2. Attention Scores ---
        # [B, 1, d_model] @ [B, d_model, N] -> [B, 1, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights) 
        
        # --- 3. Contextualization ---
        # Moltiplichiamo le values per i pesi (Broadcasting su d_vis)
        # [B, 1, N] -> transpose -> [B, N, 1] * [B, N, d_vis]
        vis_context = V * attn_weights.transpose(1, 2)
        
        # --- 4. RESIDUAL CONNECTION & NORMALIZATION ---
        # Sommiamo il contesto calcolato al segnale visivo originale (vis_x)
        # In questo modo, se l'attenzione è "rumorosa", il modello può imparare 
        # a ignorarla tenendo i pesi vicini a zero.
        out = self.norm(vis_x + vis_context)
        
        return out, attn_weights.squeeze(1)