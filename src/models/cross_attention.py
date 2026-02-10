import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model=128):
        super().__init__()
        self.q_proj = nn.Linear(d_clin, d_model)
        self.k_proj = nn.Linear(d_vis, d_model)
        self.v_proj = nn.Linear(d_vis, d_model)

        self.scale = d_model ** -0.5

    def forward(self, clin_emb, vis_emb):
        """
        clin_emb: [B, D_clin]
        vis_emb:  [B, N, D_vis]
        """
        Q = self.q_proj(clin_emb).unsqueeze(1)      # [B, 1, D]
        K = self.k_proj(vis_emb)                    # [B, N, D]
        V = self.v_proj(vis_emb)                    # [B, N, D]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)              # [B, 1, N]

        out = torch.matmul(attn, V)                 # [B, 1, D]
        return out.squeeze(1), attn.squeeze(1)      # [B, D], [B, N]