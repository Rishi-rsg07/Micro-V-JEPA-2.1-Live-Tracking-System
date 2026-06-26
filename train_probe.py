import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from downstream import AttentiveSegmentationProbe

# 1. Custom PyTorch Dataset to load your saved V-JEPA latents
class VJEPADataset(Dataset):
    def __init__(self, base_dir="dataset"):
        self.samples = []
        self.labels = []
        
        # Load background files (Class 0)
        bg_dir = os.path.join(base_dir, "background")
        if os.path.exists(bg_dir):
            for f in os.listdir(bg_dir):
                if f.endswith(".npy"):
                    self.samples.append(os.path.join(bg_dir, f))
                    self.labels.append(0)
                    
        # Load target files (Class 1)
        tgt_dir = os.path.join(base_dir, "target")
        if os.path.exists(tgt_dir):
            for f in os.listdir(tgt_dir):
                if f.endswith(".npy"):
                    self.samples.append(os.path.join(tgt_dir, f))
                    self.labels.append(1)
                    
        print(f"📦 Loaded Dataset: {self.labels.count(1)} target frames, {self.labels.count(0)} background frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the saved raw features [196, 192]
        features = np.load(self.samples[idx])
        label = self.labels[idx]
        
        # Target Generation: 
        # For background (0), all 196 patches are class 0.
        # For target (1), we assume you are sitting roughly in the center patches (rows 4-10, cols 4-10)
        spatial_target = np.zeros((14, 14), dtype=np.int64)
        if label == 1:
            spatial_target[4:10, 4:10] = 1
            
        return torch.from_numpy(features).float(), torch.from_numpy(spatial_target.flatten())

# 2. Main Training Execution Loop
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training probe on device: {device}")
    
    # Hyperparameters
    epochs = 15
    batch_size = 16
    learning_rate = 0.001
    
    # Initialize components
    dataset = VJEPADataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate probe using your exact architectural parameters
    probe = AttentiveSegmentationProbe(embed_dim=192, num_classes=2).to(device)
    probe.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print("\n🚀 Starting Training Loop...")
    for epoch in range(epochs):
        running_loss = 0.0
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device) # features: [B, 196, 192], targets: [B, 196]
            
            optimizer.zero_grad()
            
            # Forward pass through your architecture's segmentation probe
            logits = probe(features) # Output shape: [B, 196, 2]
            
            # Reshape tensors to match PyTorch CrossEntropy dimensions: [B * 196, 2] vs [B * 196]
            loss = criterion(logits.view(-1, 2), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f}")
        
    # Save the optimized weights back out safely
    torch.save(probe.state_dict(), "probe_trained.pth")
    print("\n🎯 Training complete! Optimized probe weights saved as 'probe_trained.pth'")

if __name__ == "__main__":
    train_pipeline()