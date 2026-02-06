"""Analyze energy ||pred_noise||^2 on manifold vs uniform points."""
import argparse
import numpy as np
import torch

from models import BaselineDDPM
from diffusion import DiffusionScheduler
from data import SpiralDataset


def load_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    m_hidden_dim = config.get("hidden_dim", 128)
    m_num_layers = config.get("num_layers", 3)
    m_rff_scale = config.get("rff_scale", 30.0)

    model = BaselineDDPM(
        hidden_dim=m_hidden_dim,
        num_layers=m_num_layers,
        rff_scale=m_rff_scale,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def energy_stats(model, points: torch.Tensor, t_star: int):
    device = next(model.parameters()).device
    x = points.to(device)
    t = torch.full((x.shape[0],), t_star, device=device, dtype=torch.long)
    with torch.no_grad():
        eps = model(x, t.float())
        energy = torch.sum(eps ** 2, dim=1)
    energy_np = energy.cpu().numpy()
    return {
        "mean": float(np.mean(energy_np)),
        "median": float(np.median(energy_np)),
        "p10": float(np.percentile(energy_np, 10)),
        "p90": float(np.percentile(energy_np, 90)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze energy on manifold vs uniform")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--t_star", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.model_path, args.device)

    # Manifold points (GT)
    gt = SpiralDataset(n_samples=args.n_samples, noise=0.0).data

    # Uniform points within normalized range
    uniform = (torch.rand(args.n_samples, 2) - 0.5) * 2.0

    gt_stats = energy_stats(model, gt, args.t_star)
    uni_stats = energy_stats(model, uniform, args.t_star)

    print("=== Energy Statistics (||pred_noise||^2) ===")
    print(f"t_star: {args.t_star}")
    print("GT manifold:", gt_stats)
    print("Uniform:", uni_stats)

    # Compare how often GT energy is lower than uniform
    device = args.device
    t = torch.full((args.n_samples,), args.t_star, device=device, dtype=torch.long)
    with torch.no_grad():
        gt_eps = model(gt.to(device), t.float())
        uni_eps = model(uniform.to(device), t.float())
        gt_e = torch.sum(gt_eps ** 2, dim=1)
        uni_e = torch.sum(uni_eps ** 2, dim=1)
        better_ratio = (gt_e < uni_e).float().mean().item()
    print(f"P(GT energy < Uniform energy): {better_ratio:.3f}")


if __name__ == "__main__":
    main()
