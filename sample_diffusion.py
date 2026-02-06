"""Sample points from diffusion model starting from random noise and visualize."""
import argparse
import os
import torch
import matplotlib.pyplot as plt

from models import BaselineDDPM
from diffusion import DiffusionScheduler, DDPM
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
    num_timesteps = config.get("num_timesteps", 1000)
    return model, num_timesteps


def main():
    parser = argparse.ArgumentParser(description="Sample from diffusion model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="outputs/sink_experiment")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling (faster, deterministic)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overlay_gt", action="store_true", help="Overlay GT spiral for comparison")
    parser.add_argument("--gt_noise", type=float, default=0.0, help="Noise for GT overlay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, num_timesteps = load_model(args.model_path, args.device)
    # 训练时使用 cosine schedule，这里需要保持一致
    scheduler = DiffusionScheduler(num_timesteps=num_timesteps, schedule_type='cosine', device=args.device)
    ddpm = DDPM(model, scheduler, args.device)

    with torch.no_grad():
        samples, _ = ddpm.sample(
            num_samples=args.num_samples,
            return_trajectory=False,
            use_ddim=args.use_ddim,
        )

    samples = samples.cpu().numpy()

    plt.figure(figsize=(8, 8))
    if args.overlay_gt:
        gt = SpiralDataset(n_samples=args.num_samples, noise=args.gt_noise).data.numpy()
        plt.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.15, c='black', label='GT Spiral')

    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.6, label='Samples')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    if args.overlay_gt:
        plt.legend(loc='upper right')
    out_path = os.path.join(args.output_dir, "sample_diffusion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
