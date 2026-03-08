import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model, n_heads=4, dropout=0.2, abmil=None):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # --- PROJECTIONS ---
        # Query: [B, N, d_vis]
        self.q_proj = nn.Linear(d_vis, d_model)
        # Key e Value: [B, d_clin]
        self.k_proj = nn.Linear(d_clin, d_model)
        self.v_proj = nn.Linear(d_clin, d_model)
        
        # Final projection
        # self.out_proj = nn.Linear(d_vis + d_model, d_vis)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = self.d_k ** -0.5

    def forward(self, clin_emb, vis_x):
        """
        clin_emb: [B, d_clin]
        vis_x:    [B, N, d_vis]
        """
        B, N, _ = vis_x.shape
        
        # 1. Projection and head splitting
        # Q: [B, N, d_model] -> [B, N, h, d_k] -> [B, h, N, d_k]
        Q = self.q_proj(vis_x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # K, V: [B, 1, d_model] -> [B, 1, h, d_k] -> [B, h, 1, d_k]
        clin_seq = clin_emb.unsqueeze(1) 
        K = self.k_proj(clin_seq).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(clin_seq).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Q: [B, h, N, d_k] @ K^T: [B, h, d_k, 1] -> [B, h, N, 1]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 3. Exctraction clinical context from every patch
        # [B, h, N, 1] @ V: [B, h, 1, d_k] -> [B, h, N, d_k]
        context = torch.matmul(attn_weights, V)
        
        # Re-combine heads -> [B, N, d_model]
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        # 4. Concatenation
        # combined = torch.cat([vis_x, context], dim=-1) # [B, N, d_vis + d_model]
        
        # 5. Final projection + Residual
        out = self.norm(context + clin_emb) # [B, N, d_vis]
        
        return out, attn_weights.mean(dim=1).squeeze(-1)