import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model, n_heads=4, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # --- PROIEZIONI AGGIORNATE ---
        # Query: ora derivano dalle patch visive [B, N, d_vis]
        self.q_proj = nn.Linear(d_vis, d_model)
        # Key e Value: ora derivano dalla parte clinica [B, d_clin]
        self.k_proj = nn.Linear(d_clin, d_model)
        self.v_proj = nn.Linear(d_clin, d_model)
        
        # Proiezione finale per ricombinare il contesto con la patch
        # Concateniamo il contesto clinico estratto (d_model) alla patch [d_vis]
        self.out_proj = nn.Linear(d_vis + d_model, d_vis)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_vis)
        self.scale = self.d_k ** -0.5

    def forward(self, clin_emb, vis_x):
        """
        clin_emb: [B, d_clin] -> Parte Clinica (K, V)
        vis_x:    [B, N, d_vis] -> Parte Visiva (Q)
        """
        B, N, _ = vis_x.shape
        
        # 1. Proiezioni e Split delle teste
        # Q (Visiva): [B, N, d_model] -> [B, N, h, d_k] -> [B, h, N, d_k]
        Q = self.q_proj(vis_x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # K, V (Clinica): [B, 1, d_model] -> [B, 1, h, d_k] -> [B, h, 1, d_k]
        # Usiamo unsqueeze(1) per trattare la clinica come una sequenza di lunghezza 1
        clin_seq = clin_emb.unsqueeze(1) 
        K = self.k_proj(clin_seq).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(clin_seq).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Q: [B, h, N, d_k] @ K^T: [B, h, d_k, 1] -> [B, h, N, 1]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 3. Estrazione del contesto clinico per ogni patch
        # [B, h, N, 1] @ V: [B, h, 1, d_k] -> [B, h, N, d_k]
        context = torch.matmul(attn_weights, V)
        
        # Re-combine heads -> [B, N, d_model]
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        # 4. Integrazione Residuale
        # Ora context è già [B, N, d_model], quindi combacia con vis_x [B, N, d_vis]
        # Concateniamo lungo l'ultima dimensione (le feature)
        combined = torch.cat([vis_x, context], dim=-1) # [B, N, d_vis + d_model]
        
        # Proiezione alla dimensione originale + Residual Connection + Norm
        out = self.norm(vis_x + self.out_proj(combined))
        
        # Restituiamo i pesi (media sulle teste) [B, N, 1] -> [B, N]
        return out, attn_weights.mean(dim=1).squeeze(-1)