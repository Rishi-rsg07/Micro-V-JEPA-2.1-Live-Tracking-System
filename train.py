import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import SyntheticSpatiotemporalDataset
from models import HierarchicalEncoder, MultiLevelFusionMLP, LatentPredictor, DensePredictorLoss
def train_vjepa():
    # Force CPU temporarily to isolate and eliminate driver/VRAM freeze loops
    device = torch.device("cpu") 
    print(f"🔥 Commencing Pre-Training Run on: {device}")
    
    # 1. Pipeline Entities Configuration
    dataset = SyntheticSpatiotemporalDataset(num_samples=150)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    context_encoder = HierarchicalEncoder().to(device)
    fusion_mlp = MultiLevelFusionMLP().to(device)
    predictor = LatentPredictor().to(device)
    loss_criterion = DensePredictorLoss(lambda_context=0.5)
    
    optimizer = optim.AdamW(
        list(context_encoder.parameters()) + 
        list(fusion_mlp.parameters()) + 
        list(predictor.parameters()), 
        lr=1e-3
    )
    
    print("🏃 Starting Micro-V-JEPA 2.1 Joint Pre-Training Loop...")
    for epoch in range(1, 16):
        epoch_combined, epoch_masked, epoch_dense = 0, 0, 0
        
        for batch_img, batch_target, batch_masks in dataloader:
            batch_img = batch_img.to(device)
            batch_target = batch_target.to(device)
            batch_masks = batch_masks.to(device)
            
            optimizer.zero_grad()
            
            # Step A: Pass context elements forward through structural fusion blocks
            fused_context = fusion_mlp(context_encoder(batch_img))
            
            # Step B: Generate missing pixel estimations via the Latent Predictor block
            predictions = predictor(fused_context, batch_target * batch_masks.unsqueeze(-1))
            
            # Step C: Evaluate combined Dense All-Token loss values
            total_loss, m_loss, c_loss = loss_criterion(predictions, batch_target, batch_masks)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_combined += total_loss.item()
            epoch_masked += m_loss.item()
            epoch_dense += c_loss.item()

            # Explicitly clear batch tensor references from local scope to drop references immediately
            del batch_img, batch_target, batch_masks, fused_context, predictions, total_loss
            
        # Clear out VRAM fragmentation cache at the end of every epoch milestone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch:02d}/15 | Combined Loss: {epoch_combined/len(dataloader):.5f} | "
              f"Masked Component: {epoch_masked/len(dataloader):.5f} | "
              f"Dense Ground Component: {epoch_dense/len(dataloader):.5f}")

    print("💾 Saving structural check-point parameters locally.")
    torch.save({
        'encoder': context_encoder.state_dict(),
        'fusion': fusion_mlp.state_dict()
    }, 'vjepa_backbone.pth')
    print("✅ Pre-training completed successfully!")

if __name__ == "__main__":
    train_vjepa()
    