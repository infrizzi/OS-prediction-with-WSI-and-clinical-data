import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model, n_heads=4, dropout=0.3, abmil=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.abmil = abmil

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # self.clinical_proj = nn.Linear(d_clin, d_model)

        self.q_proj = nn.Linear(d_clin, d_model)
        self.k_proj = nn.Linear(d_vis, d_model)
        self.v_proj = nn.Linear(d_vis, d_model)
        
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
        # residual = self.clinical_proj(clin_emb) # [B, d_model]

        # Q: [B, n_heads, 1, d_k]
        Q = self.q_proj(clin_emb).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        # K, V: [B, n_heads, N, d_k]
        K = self.k_proj(vis_x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(vis_x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # [B, h, 1, d_k] @ [B, h, d_k, N] -> [B, h, 1, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 3. Final context
        # [B, h, 1, N] @ [B, h, N, d_k] -> [B, h, 1, d_k]
        context = torch.matmul(attn_weights, V)
        
        # 4. Re-combine heads -> [B, 1, d_model]
        context = context.transpose(1, 2).contiguous().view(B, self.d_model)
        
        # 5. ABMIL aggregation as residual (testing to find best residual connection)
        abmil_emb, _ = self.abmil(vis_x)  # [B, d_vis]

        # 6. Norm + Residual
        out = self.norm(context + abmil_emb)
        # print(out.shape)
        
        return out, attn_weights.mean(dim=1).squeeze(1)