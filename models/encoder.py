import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """Standard lightweight Transformer Layer for self-attention feature extraction."""
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=3, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class HierarchicalEncoder(nn.Module):
    def __init__(self, embed_dim=192, depth=6):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim)) # Capable of supporting 14x14 grid tokens
        self.layers = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(depth)])
        
    def forward(self, x):
        # Adjust position tracking matrices dynamically if sequence changes
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        
        intermediate_features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Harvest intermediate states at checkpoints (Layer indices 1, 3, and 5)
            if i in [1, 3, 5]:
                intermediate_features.append(x)
                
        return intermediate_features

class MultiLevelFusionMLP(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        # Blends the stacked outputs of the 3 intermediate processing milestones
        self.fusion_projection = nn.Linear(embed_dim * 3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, intermediate_list):
        """Concats and compresses feature arrays across hidden channels."""
        stacked = torch.cat(intermediate_list, dim=-1) # [B, N, D * 3]
        fused = self.fusion_projection(stacked)        # [B, N, D]
        return self.norm(fused)