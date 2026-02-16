import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_clin, d_vis, d_model):
        super().__init__()

        # Query: clinical projection
        self.q_proj = nn.Linear(d_clin, d_model)

        # Key: visual projection
        self.k_proj = nn.Linear(d_vis, d_model)

        # Value: visual projection (with same dimension)
        self.v_proj = nn.Linear(d_vis, d_vis) 
        
        self.attn_dropout = nn.Dropout(0.3)

        # LayerNorm to stabilize training and allow residual learning
        self.norm = nn.LayerNorm(d_vis)
        
        # Scaling factor
        self.scale = d_model ** -0.5

    def forward(self, clin_emb, vis_x):
        """
        clin_emb: [B, d_clin]
        vis_x:    [B, N, d_vis]
        """
        # --- 1. Creation Q, K, V ---
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
        # Multiply values with attention weights (broadcasting on d_vis)
        # [B, 1, N] -> transpose
        # [B, N, d_vis] * [B, N, 1] -> [B, N, d_vis]
        vis_context = V * attn_weights.transpose(1, 2)
        
        # --- 4. RESIDUAL CONNECTION & NORMALIZATION ---
        # Sum context (vis_context) to original visual signal (vis_x)
        out = self.norm(vis_x + vis_context)
        
        return out, attn_weights.squeeze(1)