"""
可视化模块：绘制实验验证图表
包括采样轨迹图、模长热力图、向量场图
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from typing import List, Dict, Optional, Tuple

from diffusion import DDPM


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def plot_trajectories(self, 
                          ddpm: DDPM, 
                          num_samples: int = 20, 
                          title: str = "Sampling Trajectories",
                          save_path: str = None):
        """
        图表 1: 采样轨迹对比
        绘制从 x_T 到 x_0 的轨迹
        """
        ddpm.model.eval()
        
        with torch.no_grad():
            _, trajectory = ddpm.sample(
                num_samples=num_samples,
                return_trajectory=True,
                trajectory_interval=1
            )
        
        # 将轨迹转换为 numpy 数组: (timesteps, samples, 2)
        traj_stack = torch.stack(trajectory).cpu().numpy()
        timesteps, n_samples, _ = traj_stack.shape
        
        plt.figure(figsize=(10, 10))
        
        # 绘制每条轨迹
        for i in range(num_samples):
            # 颜色随时间变化：起点(T)浅色，终点(0)深色
            # 实际上我们通常用颜色表示方向或只是为了区分
            # 这里我们用 alpha 变化表示时间进程
            
            # 使用 cmap 来给不同轨迹上色
            color = cm.viridis(i / num_samples)
            
            plt.plot(traj_stack[:, i, 0], traj_stack[:, i, 1], 
                     alpha=0.6, linewidth=1, color=color)
            
            # 标记起点和终点
            plt.scatter(traj_stack[0, i, 0], traj_stack[0, i, 1], 
                        c='red', s=10, alpha=0.5, label='Start' if i==0 else "") # x_T
            plt.scatter(traj_stack[-1, i, 0], traj_stack[-1, i, 1], 
                        c='blue', s=20, label='End' if i==0 else "") # x_0
            
        plt.title(title)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
    def plot_norm_heatmap(self,
                          ddpm: DDPM,
                          t_value: int = 500,
                          grid_size: int = 50,
                          range_limit: float = 2.0,
                          title: str = "Norm Heatmap",
                          save_path: str = None):
        """
        图表 2: 模长热力图
        在 2D 平面上绘制固定时刻的预测噪声模长
        """
        ddpm.model.eval()
        
        # 创建网格
        x = np.linspace(-range_limit, range_limit, grid_size)
        y = np.linspace(-range_limit, range_limit, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        grid_points = torch.FloatTensor(np.stack([xx.flatten(), yy.flatten()], axis=1)).to(self.device)
        t_batch = torch.full((grid_points.shape[0],), t_value, device=self.device).float()
        
        with torch.no_grad():
            if hasattr(ddpm.model, 'get_prediction_components'):
                components = ddpm.model.get_prediction_components(grid_points, t_batch)
                norm_vals = components['norm'].cpu().numpy()
            else:
                epsilon_pred = ddpm.model(grid_points, t_batch)
                norm_vals = torch.norm(epsilon_pred, dim=-1).cpu().numpy()
        
        norm_grid = norm_vals.reshape(grid_size, grid_size)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(norm_grid, 
                        extent=[-range_limit, range_limit, -range_limit, range_limit],
                        origin='lower', cmap='plasma', vmin=0, vmax=2.5)
        plt.colorbar(im, label=f'Predicted Norm (t={t_value})')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(False)
        
        # 画个圈表示可能的流形范围
        circle = plt.Circle((0, 0), 1.0, fill=False, color='white', linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Norm heatmap saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
    def plot_vector_field(self,
                          ddpm: DDPM,
                          t_value: int = 500,
                          grid_size: int = 20,
                          range_limit: float = 2.0,
                          dataset=None,
                          title: str = "Vector Field",
                          save_path: str = None):
        """
        图表 3: 向量场图
        绘制预测噪声的方向场（忽略模长或归一化）
        """
        ddpm.model.eval()
        
        # 创建网格
        x = np.linspace(-range_limit, range_limit, grid_size)
        y = np.linspace(-range_limit, range_limit, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        grid_points = torch.FloatTensor(np.stack([xx.flatten(), yy.flatten()], axis=1)).to(self.device)
        t_batch = torch.full((grid_points.shape[0],), t_value, device=self.device).float()
        
        with torch.no_grad():
            if hasattr(ddpm.model, 'get_prediction_components'):
                components = ddpm.model.get_prediction_components(grid_points, t_batch)
                # 使用分离出的方向（已经是单位向量）
                vectors = components['direction'].cpu().numpy()
                # 为了可视化反向过程，我们通常想看的是它想把点推向哪里
                # x_{t-1} ∝ x_t - c * \epsilon
                # 所以把 \epsilon 的反方向画出来，表示去噪方向
                vectors = -vectors 
            else:
                epsilon_pred = ddpm.model(grid_points, t_batch)
                vectors = epsilon_pred.cpu().numpy()
                vectors = -vectors # 去噪方向
                # 归一化以便只展示方向
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / (norms + 1e-8)
        
        u = vectors[:, 0].reshape(grid_size, grid_size)
        v = vectors[:, 1].reshape(grid_size, grid_size)
        
        plt.figure(figsize=(10, 10))
        
        # 如果有数据集，先画个背景
        if dataset is not None:
            data = dataset.data.numpy()
            # 随机采样一部分数据点
            indices = np.random.choice(len(data), 2000, replace=False)
            plt.scatter(data[indices, 0], data[indices, 1], s=1, c='gray', alpha=0.3, label='Data Manifold')
        
        # 绘制向量场
        plt.quiver(xx, yy, u, v, color='blue', scale=25, alpha=0.8, width=0.003)
        
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-range_limit, range_limit)
        plt.ylim(-range_limit, range_limit)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Vector field saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def compare_models(self, 
                       baseline_ddpm: DDPM, 
                       decoupled_ddpm: DDPM,
                       output_dir: str = 'outputs/comparison',
                       t_list: List[int] = [100, 500, 900]):
        """
        生成两组模型的对比图
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 轨迹对比
        self.plot_trajectories(baseline_ddpm, title="Baseline Trajectories", 
                               save_path=os.path.join(output_dir, 'traj_baseline.png'))
        self.plot_trajectories(decoupled_ddpm, title="Decoupled Trajectories", 
                               save_path=os.path.join(output_dir, 'traj_decoupled.png'))
        
        # 2. 对不同 t 的热力图对比
        for t in t_list:
            self.plot_norm_heatmap(baseline_ddpm, t_value=t, title=f"Baseline Norm (t={t})",
                                   save_path=os.path.join(output_dir, f'norm_baseline_t{t}.png'))
            self.plot_norm_heatmap(decoupled_ddpm, t_value=t, title=f"Decoupled Norm (t={t})",
                                   save_path=os.path.join(output_dir, f'norm_decoupled_t{t}.png'))
            
        # 3. 向量场对比
        for t in t_list:
            self.plot_vector_field(baseline_ddpm, t_value=t, title=f"Baseline Vector Field (t={t})",
                                   save_path=os.path.join(output_dir, f'vec_baseline_t{t}.png'))
            self.plot_vector_field(decoupled_ddpm, t_value=t, title=f"Decoupled Vector Field (t={t})",
                                   save_path=os.path.join(output_dir, f'vec_decoupled_t{t}.png'))


if __name__ == "__main__":
    # 测试代码
    print("This module provides visualization tools.")
