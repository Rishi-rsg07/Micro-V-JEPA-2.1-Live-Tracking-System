import torch
import torch.nn as nn

class AttentiveSegmentationProbe(nn.Module):
    def __init__(self, embed_dim=192, num_classes=2):
        super().__init__()
        # Standard Cross-Attention block translating encoder states to classification targets
        self.query_layer = nn.Parameter(torch.zeros(1, 196, embed_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=3, batch_first=True)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, fused_encoder_features):
        batch_size = fused_encoder_features.shape[0]
        q = self.query_layer.expand(batch_size, -1, -1)
        
        # Cross-examine encoder representations using standard attention matrix query routines
        attention_out, _ = self.cross_attention(query=q, key=fused_encoder_features, value=fused_encoder_features)
        logits = self.classifier(attention_out) # Shape dimensions output match: [B, N, NumClasses]
        return logits 