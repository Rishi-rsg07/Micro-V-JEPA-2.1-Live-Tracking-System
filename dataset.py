import torch
from torch.utils.data import Dataset

class SyntheticSpatiotemporalDataset(Dataset):
    def __init__(self, num_samples=30, num_patches=196): # Reduced from 100 to 30
        self.num_samples = num_samples
        self.num_patches = num_patches

    def __getitem__(self, idx):
        simulated_image_tokens = torch.randn(self.num_patches, 192)
        target_encoder_labels = simulated_image_tokens + torch.randn(self.num_patches, 192) * 0.1
        random_mask_map = torch.bernoulli(torch.full((self.num_patches,), 0.4))
        return simulated_image_tokens, target_encoder_labels, random_mask_map

    def __len__(self):
        return self.num_samples