"""Quick script to check model training status"""
import torch

cp = torch.load('outputs/baseline_spiral/checkpoints/checkpoint_final.pt', map_location='cpu')
config = cp.get('config', {})
loss = cp.get('history', {}).get('loss', [])

print("=== Model Config ===")
for k, v in config.items():
    print(f"  {k}: {v}")

print(f"\n=== Training History ===")
print(f"  Epochs trained: {len(loss)}")
if loss:
    print(f"  First epoch loss: {loss[0]:.6f}")
    print(f"  Last epoch loss: {loss[-1]:.6f}")
    print(f"  Min loss: {min(loss):.6f} (at epoch {loss.index(min(loss))+1})")
    
    # Check if model converged
    if len(loss) > 10:
        recent_avg = sum(loss[-10:]) / 10
        early_avg = sum(loss[:10]) / 10
        print(f"  Early avg (first 10): {early_avg:.6f}")
        print(f"  Recent avg (last 10): {recent_avg:.6f}")
        print(f"  Improvement ratio: {early_avg / recent_avg:.2f}x")

# Check data range
from data import SpiralDataset
ds = SpiralDataset(n_samples=5000, noise=0.0)
data = ds.data.numpy()
print(f"\n=== Data Range ===")
print(f"  X range: [{data[:,0].min():.4f}, {data[:,0].max():.4f}]")
print(f"  Y range: [{data[:,1].min():.4f}, {data[:,1].max():.4f}]")
print(f"  Max abs: {abs(data).max():.4f}")
