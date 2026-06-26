import torch
import torch.nn as nn

class MultiModalTokenizer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 2.D Spatial Convolution for Static Images
        self.proj_2d = nn.Conv2d(
            in_channels=3, out_channels=embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 3D Spatiotemporal Convolution for Video Tubelets (kernel covering 2 frames)
        self.proj_3d = nn.Conv3d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )
        
        # Distinct, learnable modality indicators to anchor spatial positioning
        self.modality_embed_image = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embed_video = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x, modality="image"):
        """
        Args:
            x: Input tensor [B, C, H, W] for image, [B, C, T, H, W] for video
        Returns:
            Tokens: Flattened embed array ready for Transformer execution [B, N, D]
        """
        if modality == "image":
            # Input Shape: [B, 3, H, W]
            tokens = self.proj_2d(x) # [B, D, H/P, W/P]
            tokens = tokens.flatten(2).transpose(1, 2) # [B, N, D]
            tokens = tokens + self.modality_embed_image
        elif modality == "video":
            # Input Shape: [B, 3, T, H, W]
            tokens = self.proj_3d(x) # [B, D, T_tokens, H/P, W/P]
            tokens = tokens.flatten(2).transpose(1, 2) # [B, N, D]
            tokens = tokens + self.modality_embed_video
        else:
            raise ValueError(f"Unknown modality target context flag: {modality}")
            
        return tokens