"""
Sink Point Optimization Experiment (sink_optimization.py)

实验目标：
1. 生成初始样本点云 (Standard Sampling)
2. 对样本点进行 Test-time Optimization (Sink Point Seeking)
   Objective: Minimize ||epsilon_theta(x, t*)||^2
3. 对比分析 Score Field (预测场) 和 Gradient Field (优化场)

架构设计：
- SinkOptimizer: 核心逻辑类，负责加载模型、采样、优化
- FieldAnalyzer: 分析工具类，负责计算两种向量场
- SinkVisualizer: 可视化类，负责绘制四种核心图表
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from tqdm import tqdm

from models import BaselineDDPM
from diffusion import DiffusionScheduler
from data import SpiralDataset # Import dataset for GT visualization

class FieldAnalyzer:
    """场分析器：计算 Score Field 和 Gradient Field"""
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        
    def get_score_field(self, x: torch.Tensor, t_value: int) -> torch.Tensor:
        """
        计算 Score Field (预测场)
        V_score = -epsilon_theta(x, t)
        """
        self.model.eval()
        t_batch = torch.full((x.shape[0],), t_value, device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            epsilon_pred = self.model(x, t_batch.float())
            
        return -epsilon_pred
    
    def get_gradient_field(self, x: torch.Tensor, t_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 Gradient Field (优化场)
        V_grad = -grad_x ||epsilon_theta(x, t)||^2
        
        Returns:
            grad_x: 梯度向量
            energy: 能量值 ||epsilon||^2
        """
        self.model.eval() # 确保模型处于 eval 模式 (Dropout/BN 固定)
        
        # 必须 clone 并开启梯度，因为 x 是输入
        x_in = x.clone().detach().requires_grad_(True)
        t_batch = torch.full((x_in.shape[0],), t_value, device=self.device, dtype=torch.long)
        
        # 前向传播
        epsilon_pred = self.model(x_in, t_batch.float())
        
        # 计算能量: E = ||epsilon||^2
        energy = torch.sum(epsilon_pred ** 2, dim=1)
        
        # 计算对 x 的梯度
        grad_x = torch.autograd.grad(outputs=energy.sum(), inputs=x_in)[0]
        
        return -grad_x, energy.detach()


class SinkOptimizer:
    """Sink 优化实验核心逻辑"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_timesteps: int = 1000,
                 hidden_dim: int = 128):
        self.device = device
        
        # 1. 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 尝试从 checkpoint 配置中读取模型参数，如果不存在则使用默认值
        config = checkpoint.get('config', {})
        m_hidden_dim = config.get('hidden_dim', hidden_dim)
        m_num_layers = config.get('num_layers', 3) # 默认为 main.py 的设定 (3) 而非 models.py (4)
        
        print(f"Loading model with hidden_dim={m_hidden_dim}, num_layers={m_num_layers}...")
        
        self.model = BaselineDDPM(
            hidden_dim=m_hidden_dim,
            num_layers=m_num_layers
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 2. 调度器
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps, device=device)
        
        # 3. 分析器
        self.analyzer = FieldAnalyzer(self.model, device)
        
    def sample_initial_points(self, num_samples: int = 500) -> torch.Tensor:
        """生成初始采样点"""
        print("Sampling initial points...")
        # 使用 DDIM 采样以获得更确定性的初始分布
        samples, _ = self.scheduler.sample(
            self.model, 
            num_samples=num_samples, 
            return_trajectory=False, 
            use_ddim=True
        )
        return samples
        
    def optimize_points(self, 
                        init_points: torch.Tensor, 
                        t_star: int = 10, 
                        steps: int = 50, 
                        lr: float = 0.01) -> np.ndarray:
        """
        对点云进行 Sink Point 优化
        
        Returns:
            trajectory: shape (steps+1, num_samples, 2) numpy array
        """
        print(f"Optimizing points towards sink (t={t_star}, steps={steps}, lr={lr})...")
        
        x = init_points.clone().detach()
        trajectory = [x.cpu().numpy()]
        
        for _ in tqdm(range(steps)):
            # 直接利用 analyzer 计算梯度
            # V_grad = -grad E，所以更新公式是 x = x + lr * V_grad
            grad_direction, _ = self.analyzer.get_gradient_field(x, t_star)
            
            # 更新 x: 沿着负梯度方向移动 (Minimize Energy)
            # x_{k+1} = x_k + lr * (-grad E)
            x = x + lr * grad_direction
            
            trajectory.append(x.cpu().numpy())
            
        return np.array(trajectory)


class SinkVisualizer:
    """可视化模块"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_sink_trajectory(self, 
                             trajectory: np.ndarray, 
                             gt_data: np.ndarray = None,
                             title: str = "Sink Optimization Trajectory"):
        """图1: 优化轨迹图"""
        steps, num_samples, _ = trajectory.shape
        
        plt.figure(figsize=(10, 10))

        # 绘制 GT (Ground Truth Manifold)
        if gt_data is not None:
            plt.scatter(gt_data[:, 0], gt_data[:, 1], s=2, c='black', alpha=0.15, label='Theoretical Manifold', zorder=0)
        
        # 绘制终点 (Sink Points)
        plt.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], 
                   c='blue', s=10, label='Optimized (Sink)', zorder=3)
        
        # 绘制起点 (Initial Samples)
        plt.scatter(trajectory[0, :, 0], trajectory[0, :, 1], 
                   c='red', s=10, alpha=0.5, label='Initial Sample', zorder=2)
        
        # 绘制轨迹
        for i in range(num_samples):
             plt.plot(trajectory[:, i, 0], trajectory[:, i, 1], 
                     alpha=0.1, color='gray', linewidth=0.5, zorder=1)
             
        plt.title(title)
        plt.legend()
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, "1_sink_trajectory.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_fields(self, 
                    optimizer: SinkOptimizer, 
                    t_eval: int, 
                    gt_data: np.ndarray = None,
                    grid_size: int = 25, 
                    range_limit: float = 1.5):
        """
        绘制图2, 3, 4: 场分析图
        """
        # 创建网格
        x = np.linspace(-range_limit, range_limit, grid_size)
        y = np.linspace(-range_limit, range_limit, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = torch.FloatTensor(np.stack([xx.flatten(), yy.flatten()], axis=1)).to(optimizer.device)
        
        # 计算两种场
        score_vectors = optimizer.analyzer.get_score_field(grid_points, t_eval).cpu().numpy()
        grad_vectors, energy_vals = optimizer.analyzer.get_gradient_field(grid_points, t_eval)
        grad_vectors = grad_vectors.cpu().numpy()
        energy_vals = energy_vals.cpu().numpy().reshape(grid_size, grid_size)
        
        # 归一化用于可视化方向 (可选，或者只画方向)
        norm_score = np.linalg.norm(score_vectors, axis=1, keepdims=True) + 1e-8
        norm_grad = np.linalg.norm(grad_vectors, axis=1, keepdims=True) + 1e-8
        
        # 筛选: 忽略梯度过小的点 (因为已经是 Sink Point 了，方向无意义)
        grad_threshold = np.percentile(norm_grad, 15) # 忽略最小的 15%
        mask = (norm_grad > grad_threshold).flatten()
        
        score_u, score_v = (score_vectors / norm_score).T.reshape(2, grid_size, grid_size)
        grad_u, grad_v = (grad_vectors / norm_grad).T.reshape(2, grid_size, grid_size)
        
        # 应用掩码到 Gradient Field (只显示有意义的梯度)
        # 很多绘图库不支持带 mask 的 quiver 直接输入，我们可以把被 mask 的位置设为 NaN
        grad_u_masked = grad_u.copy()
        grad_v_masked = grad_v.copy()
        flat_mask = mask.reshape(grid_size, grid_size)
        grad_u_masked[~flat_mask] = np.nan
        grad_v_masked[~flat_mask] = np.nan
        
        # --- 图2: Score Field ---
        plt.figure(figsize=(10, 10))
        if gt_data is not None:
            plt.scatter(gt_data[:, 0], gt_data[:, 1], s=1, c='black', alpha=0.1, zorder=0)
            
        plt.quiver(xx, yy, score_u, score_v, color='green', scale=30, headwidth=3, alpha=0.8, zorder=1)
        plt.title(f"Score Field (Diffusion Direction) at t={t_eval}")
        plt.xlim(-range_limit, range_limit)
        plt.ylim(-range_limit, range_limit)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "2_score_field.png"), dpi=150)
        plt.close()
        
        # --- 图3: Gradient Field (Enhanced with Energy Background) ---
        plt.figure(figsize=(12, 10))
        
        # 0. 绘制 GT (稍微显眼一点，因为背景是蓝色的)
        if gt_data is not None:
            plt.scatter(gt_data[:, 0], gt_data[:, 1], s=1, c='black', alpha=0.2, zorder=1)

        # 1. 绘制能量背景 (Log scale for better visibility of valleys)
        plt.imshow(np.log1p(energy_vals), extent=[-range_limit, range_limit, -range_limit, range_limit],
                   origin='lower', cmap='Blues', alpha=0.6, zorder=0)
        plt.colorbar(label='Log Energy log(1 + ||epsilon||^2)')
        
        # 2. 绘制梯度场 (Masked)
        plt.quiver(xx, yy, grad_u_masked, grad_v_masked, color='purple', scale=30, headwidth=3, alpha=0.9, zorder=2)
        
        plt.title(f"Gradient Field (Sink Attraction) at t={t_eval}\nBackground: Potential Energy Landscape")
        plt.xlim(-range_limit, range_limit)
        plt.ylim(-range_limit, range_limit)
        plt.grid(False) # Turn off grid to see heatmap better
        plt.savefig(os.path.join(self.output_dir, "3_gradient_field.png"), dpi=150)
        plt.close()
        
        # --- 图4: Overlay Comparison ---
        plt.figure(figsize=(12, 12))
        
        # 0. 绘制 GT
        if gt_data is not None:
            plt.scatter(gt_data[:, 0], gt_data[:, 1], s=2, c='black', alpha=0.15, label='Theoretical Manifold', zorder=0)
        
        # 为了对比清晰，减少密度
        step = 1
        plt.quiver(xx[::step, ::step], yy[::step, ::step], 
                  score_u[::step, ::step], score_v[::step, ::step], 
                  color='green', alpha=0.4, scale=35, label='Score (Diffusion)', width=0.003)
                  
        plt.quiver(xx[::step, ::step], yy[::step, ::step], 
                  grad_u_masked[::step, ::step], grad_v_masked[::step, ::step], 
                  color='purple', alpha=0.9, scale=35, label='Gradient (Sink)', width=0.003) # Gradient 更突出
        
        # 标出梯度极小的区域 (Sink Points)
        sink_points_mask = ~flat_mask
        plt.scatter(xx[sink_points_mask], yy[sink_points_mask], s=5, c='red', marker='x', alpha=0.5, label='Local Minima (Sink)')
        
        # 画个圈做参考
        circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
        
        plt.title(f"Field Comparison: Score vs Gradient at t={t_eval}\nPurple arrows show optimization path, Red 'x' are potential sinks")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "4_field_comparison.png"), dpi=150)
        plt.close()
        print("Saved field plots.")


def main():
    parser = argparse.ArgumentParser(description="Sink Point Optimization Analysis")
    parser.add_argument('--model_path', type=str, required=True, help='Path to baseline model checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='outputs/sink_experiment')
    parser.add_argument('--t_star', type=int, default=5, help='Time step to freeze for optimization')
    parser.add_argument('--opt_steps', type=int, default=100, help='Optimization steps')
    parser.add_argument('--lr', type=float, default=0.01, help='Optimization learning rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 初始化
    optimizer = SinkOptimizer(args.model_path, args.device, hidden_dim=args.hidden_dim)
    visualizer = SinkVisualizer(args.output_dir)
    
    # 0. 准备 Ground Truth 数据用于可视化
    # 假设我们知道这是 spiral 数据集 (因为模型文件名里有，或者参数指定)
    # 为了通用性，还是生成一下 Spiral 的 GT。如果不想硬编码，可以加个参数，但为了简单起见，且用户主要跑 spiral，直接生成即可。
    gt_dataset = SpiralDataset(n_samples=5000, noise=0.0) # 0 噪声不仅是流形骨架
    gt_data = gt_dataset.data.numpy()
    
    # 1. 初始采样
    # 增加点数以获得更密集的分布，便于观察流形吸附效果
    init_samples = optimizer.sample_initial_points(num_samples=args.num_samples)
    
    # 2. Sink 优化
    trajectory = optimizer.optimize_points(init_samples, t_star=args.t_star, steps=args.opt_steps, lr=args.lr)
    
    # 3. 绘制轨迹
    visualizer.plot_sink_trajectory(trajectory, gt_data=gt_data, title=f"Sink Optimization (t={args.t_star})")
    
    # 4. 绘制场分析
    visualizer.plot_fields(optimizer, args.t_star, gt_data=gt_data)
    
    print("Experiment finished.")

if __name__ == "__main__":
    main()
