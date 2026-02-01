"""
实验运行入口：运行完整的对比实验
"""

import os
import torch
import argparse
from datetime import datetime

from train import train_both_models
from visualization import Visualizer
from data import get_dataloader

def run_experiment(args):
    """
    运行完整的对比实验
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join('outputs', f'comparison_{args.dataset}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Starting Experiment: {args.dataset}")
    print(f"Output Directory: {exp_dir}")
    print("=" * 60)
    
    # 1. 训练两个模型
    baseline_trainer, decoupled_trainer = train_both_models(
        num_epochs=args.epochs,
        dataset_type=args.dataset,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rff_scale=args.rff_scale,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_dir=args.lambda_dir,
        lambda_norm=args.lambda_norm,
        device=args.device
    )
    
    # 2. 可视化对比
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Apply EMA weights if available for visualization
    if baseline_trainer.use_ema:
        print("Using EMA weights for Baseline visualization...")
        # Note: 'final' checkpoint was saved WITH EMA weights applied in train.py
        pass 
        
    if decoupled_trainer.use_ema:
        print("Using EMA weights for Decoupled visualization...")
        pass

    viz = Visualizer(device=args.device)
    
    # 比较模型
    # 选择三个代表性的时刻：接近纯噪声(900), 中间状态(500), 接近数据(100)
    t_list = [int(args.num_timesteps * 0.9), 
              int(args.num_timesteps * 0.5), 
              int(args.num_timesteps * 0.1)]
    
    viz.compare_models(
        baseline_trainer.ddpm,
        decoupled_trainer.ddpm,
        output_dir=os.path.join(exp_dir, 'figures'),
        t_list=t_list
    )
    
    # 3. 额外探究：OOD 测试
    # (可选)
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print(f"Check results in: {exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Diffusion Model Comparison Experiment')
    
    # 实验设置
    parser.add_argument('--dataset', type=str, default='spiral', 
                        choices=['spiral', 'swiss_roll'], help='Dataset type')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of training epochs (suggest 1000 for full quality)')
    parser.add_argument('--num_timesteps', type=int, default=1000, 
                        help='Number of diffusion steps (suggest 1000 for quality)')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of residual layers')
    parser.add_argument('--rff_scale', type=float, default=30.0, help='Scale for Gaussian Fourier Features (higher = more high frequency)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    
    # Loss 参数
    parser.add_argument('--lambda_dir', type=float, default=1.0, help='Weight for direction loss')
    parser.add_argument('--lambda_norm', type=float, default=0.5, help='Weight for norm loss')
    
    args = parser.parse_args()
    
    run_experiment(args)
