"""
扩散过程模块：定义前向扩散和反向采样过程
包括 Beta Schedule 和采样算法
"""

import torch
import torch.nn as nn
import copy
from typing import Tuple, Optional, List, Dict
import numpy as np


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Standard practice in Image Diffusion and Diffusion Policy.
    """
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        # Create a shadow map of parameters
        # Use a plain dict instead of nn.ModuleDict because parameters with dots in names
        # cannot be keys in nn.ParameterDict, and nn.ModuleDict expects Modules.
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        
        self.backup = {}
        
    def update(self, model: nn.Module):
        """Update shadow parameters"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # new_average = (1.0 - decay) * param + decay * shadow
                    # Note: Usually EMA is shadow = decay * shadow + (1-decay) * param
                    # The formula used here is:
                    # shadow = (1 - decay) * param + decay * shadow
                    # which is standard EMA update.
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name].copy_(new_average)

    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to the model (and backup original)"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original parameters from backup"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

class DiffusionScheduler:
    """
    扩散过程调度器
    管理 beta, alpha, alpha_bar 等参数
    """
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear',
                 device: str = 'cpu'):
        """
        Args:
            num_timesteps: 扩散步数 T
            beta_start: beta 起始值
            beta_end: beta 结束值
            schedule_type: 'linear' 或 'cosine'
            device: 计算设备
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 计算 beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule(num_timesteps, device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # 计算 alpha 和累积 alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])
        
        # 用于前向过程的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于后向采样的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_schedule(self, num_timesteps: int, device: str) -> torch.Tensor:
        """Cosine beta schedule"""
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def to(self, device: str):
        """移动到指定设备"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        从参数数组中提取对应时间步的值
        
        Args:
            a: 参数数组, shape (T,)
            t: 时间步, shape (batch_size,)
            x_shape: 输入形状
        
        Returns:
            提取的值，broadcast 到正确的形状
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程：q(x_t | x_0)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x_0: 原始数据, shape (batch_size, 2)
            t: 时间步, shape (batch_size,)
            noise: 噪声（可选），如果不提供则生成随机噪声
        
        Returns:
            x_t: 加噪后的数据
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def p_sample(self, 
                 model: nn.Module, 
                 x_t: torch.Tensor, 
                 t: torch.Tensor,
                 clip_denoised: bool = True) -> torch.Tensor:
        """
        反向采样一步：p(x_{t-1} | x_t)
        
        Args:
            model: 噪声预测模型
            x_t: 当前时刻的样本
            t: 时间步
            clip_denoised: 是否裁剪去噪后的值
        
        Returns:
            x_{t-1}: 上一时刻的样本
        """
        # 获取模型预测的噪声
        with torch.no_grad():
            epsilon_pred = model(x_t, t.float())
        
        # 计算去噪后的均值
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # x_{t-1} 的均值
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha_bar * epsilon_pred)
        
        if clip_denoised:
            mean = torch.clamp(mean, -1.5, 1.5)
        
        # 添加噪声（除了 t=0）
        if t[0] > 0:
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample(self,
                   model: nn.Module,
                   num_samples: int,
                   return_trajectory: bool = False,
                   trajectory_interval: int = 50,
                   eta: float = 0.0) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        DDIM Sampling (Deterministic if eta=0)
        More consistent with Diffusion Policy inference.
        
        Args:
            eta: 0.0 for deterministic DDIM, 1.0 for DDPM
        """
        model.eval()
        x = torch.randn(num_samples, 2, device=self.device)
        
        trajectory = [] if return_trajectory else None
        if return_trajectory: trajectory.append(x.clone())
        
        # DDIM uses a subset of steps usually, but here we can use all for quality
        # Or you can implement strided sampling (e.g. 50 steps)
        # For toy exp, we iterate all T for simplicity or use the same T schedule
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # 1. Predict Noise
            epsilon_pred = model(x, t_batch.float())
            
            # 2. Get Constants
            alpha_bar = self._extract(self.alphas_cumprod, t_batch, x.shape)
            alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t_batch, x.shape)
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
            
            # 3. Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * epsilon_pred) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.5, 1.5) # Dynamic thresholding-like clip
            
            # 4. Direction to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon_pred
            
            # 5. Update
            noise = torch.randn_like(x) if eta > 0 else 0
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            
            x = x_prev
            
            if return_trajectory and t % trajectory_interval == 0:
                trajectory.append(x.clone())
                
        if return_trajectory: trajectory.append(x.clone())
        
        return x, trajectory

    @torch.no_grad()
    def sample(self, 
               model: nn.Module,
               num_samples: int,
               return_trajectory: bool = False,
               trajectory_interval: int = 50,
               use_ddim: bool = True) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Unified sampling interface
        """
        if use_ddim:
            return self.ddim_sample(model, num_samples, return_trajectory, trajectory_interval, eta=0.0)
            
        model.eval()
        
        # 从标准高斯分布采样 x_T
        x = torch.randn(num_samples, 2, device=self.device)
        
        trajectory = [] if return_trajectory else None
        
        if return_trajectory:
            trajectory.append(x.clone())
        
        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
            
            if return_trajectory and t % trajectory_interval == 0:
                trajectory.append(x.clone())
        
        if return_trajectory:
            trajectory.append(x.clone())
        
        return x, trajectory
    
    @torch.no_grad()
    def sample_with_components(self,
                                model: nn.Module,
                                num_samples: int,
                                return_trajectory: bool = False,
                                trajectory_interval: int = 50) -> Dict:
        """
        采样并记录详细的组件信息（用于可视化分析）
        
        Returns:
            dict: {
                'samples': 最终样本,
                'trajectory': 轨迹列表,
                'norms': 每步的模长,
                'directions': 每步的方向
            }
        """
        model.eval()
        
        x = torch.randn(num_samples, 2, device=self.device)
        
        trajectory = [x.clone()]
        norms_history = []
        directions_history = []
        timesteps_history = []
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # 获取模型预测的组件
            if hasattr(model, 'get_prediction_components'):
                components = model.get_prediction_components(x, t_batch.float())
                epsilon_pred = components['epsilon']
                norms_history.append(components['norm'].mean().item())
                directions_history.append(components['direction'].clone())
            else:
                epsilon_pred = model(x, t_batch.float())
                norm = torch.norm(epsilon_pred, dim=-1)
                norms_history.append(norm.mean().item())
            
            timesteps_history.append(t)
            
            # 采样步骤
            sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t_batch, x.shape)
            beta = self._extract(self.betas, t_batch, x.shape)
            sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            
            mean = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha_bar * epsilon_pred)
            mean = torch.clamp(mean, -1.5, 1.5)
            
            if t > 0:
                posterior_var = self._extract(self.posterior_variance, t_batch, x.shape)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * noise
            else:
                x = mean
            
            if t % trajectory_interval == 0:
                trajectory.append(x.clone())
        
        return {
            'samples': x,
            'trajectory': trajectory,
            'norms': norms_history,
            'timesteps': timesteps_history
        }


class DDPM:
    """
    DDPM 训练和采样的封装类
    """
    
    def __init__(self,
                 model: nn.Module,
                 scheduler: DiffusionScheduler,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.scheduler = scheduler.to(device)
        self.device = device
    
    def train_step(self, x_0: torch.Tensor, loss_fn) -> Tuple[torch.Tensor, Dict]:
        """
        执行一个训练步
        
        Args:
            x_0: 原始数据
            loss_fn: 损失函数
        
        Returns:
            loss: 损失值
            info: 额外信息字典
        """
        batch_size = x_0.shape[0]
        
        # Importance Sampling: 对低 t 区域进行更多采样
        # 使用 sqrt 分布让模型在接近数据的时间步获得更多训练
        # t ~ sqrt(Uniform(0, T^2)) 等价于对小 t 有更高的采样概率
        u = torch.rand(batch_size, device=self.device)
        t = (u * self.scheduler.num_timesteps).long()
        t = torch.clamp(t, 0, self.scheduler.num_timesteps - 1)
        
        # 前向扩散
        noise = torch.randn_like(x_0)
        x_t, _ = self.scheduler.q_sample(x_0, t, noise)
        
        # 模型预测
        epsilon_pred = self.model(x_t, t.float())
        
        # 计算损失
        loss, loss_info = loss_fn(epsilon_pred, noise, self.model, x_t, t.float())
        
        return loss, loss_info
    
    def sample(self, num_samples: int, use_ddim: bool = True, **kwargs):
        """采样"""
        return self.scheduler.sample(self.model, num_samples, use_ddim=use_ddim, **kwargs)
    
    def sample_with_components(self, num_samples: int, **kwargs):
        """带组件信息的采样"""
        return self.scheduler.sample_with_components(self.model, num_samples, **kwargs)


if __name__ == "__main__":
    from models import BaselineDDPM, NormDecoupledModel
    
    # 测试调度器
    print("=" * 50)
    print("Testing Diffusion Scheduler")
    print("=" * 50)
    
    scheduler = DiffusionScheduler(num_timesteps=1000)
    print(f"Number of timesteps: {scheduler.num_timesteps}")
    print(f"Beta range: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    print(f"Alpha_bar range: [{scheduler.alphas_cumprod[-1]:.6f}, {scheduler.alphas_cumprod[0]:.6f}]")
    
    # 测试前向扩散
    x_0 = torch.randn(32, 2)
    t = torch.randint(0, 1000, (32,))
    x_t, noise = scheduler.q_sample(x_0, t)
    print(f"\nForward diffusion test:")
    print(f"x_0 shape: {x_0.shape}, x_t shape: {x_t.shape}")
    
    # 测试采样
    print("\n" + "=" * 50)
    print("Testing Sampling")
    print("=" * 50)
    
    model = BaselineDDPM()
    samples, trajectory = scheduler.sample(model, num_samples=100, return_trajectory=True)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Trajectory length: {len(trajectory)}")
