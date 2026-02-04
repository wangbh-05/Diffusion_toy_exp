"""
模型架构模块：定义 Backbone、Baseline DDPM 和 Norm-Decoupled 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 tensor, shape (batch_size,)
        
        Returns:
            time embedding, shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    带残差连接的 MLP 块，对齐 Modern Diffusion 架构
    Structure: Input -> GroupNorm -> SiLU -> Linear -> GroupNorm -> SiLU -> Linear
    Time Embedding 注入在中间
    """
    
    def __init__(self, in_dim: int, out_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        
        # 模拟 Image Diffusion 的 GroupNorm (num_groups=32 is standard, but for small dim we use 8)
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_dim // 4), num_channels=in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        
        self.time_proj = nn.Linear(time_dim, out_dim)
        
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_dim // 4), num_channels=out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C)
        t_emb: (B, T_dim)
        """
        h = x
        
        # Block 1
        h = self.norm1(h)
        h = self.act(h)
        h = self.fc1(h)
        
        # Add time embedding (Broadcasting like in U-Net)
        h = h + self.time_proj(self.act(t_emb))
        
        # Block 2
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        
        return h + self.residual_proj(x)


class Backbone(nn.Module):
    """
    通用骨干网络：ResNet-style MLP
    用于提取坐标和时间的联合特征
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 time_dim: int = 64,
                 num_layers: int = 4,
                 rff_scale: float = 30.0): # Default increased to 30.0 and made configurable
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 坐标嵌入 (Gaussian Fourier Features)
        self.coord_embed = GaussianFourierProjection(hidden_dim, scale=rff_scale)
        
        # 输入投影 (从傅里叶特征维度映射到隐藏层维度)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_dim)
            for _ in range(num_layers)
        ])
        
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入坐标, shape (batch_size, 2)
            t: 时间步, shape (batch_size,)
        
        Returns:
            特征向量, shape (batch_size, hidden_dim)
        """
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 坐标嵌入
        x_emb = self.coord_embed(x)
        h = self.input_proj(x_emb)
        
        # 通过残差块
        for res_block in self.res_blocks:
            h = res_block(h, t_emb)
        
        return h


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for coordinating inputs."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        # 固定随机种子以保证可复现性
        rng = torch.Generator().manual_seed(42)
        self.W = nn.Parameter(torch.randn(embed_dim // 2, 2, generator=rng) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x @ self.W.T * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BaselineDDPM(nn.Module):
    """
    标准 DDPM 模型
    直接预测噪声向量 ε ∈ R²
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 time_dim: int = 64,
                 num_layers: int = 4,
                 rff_scale: float = 30.0):
        super().__init__()
        
        self.backbone = Backbone(input_dim, hidden_dim, time_dim, num_layers, rff_scale=rff_scale)
        
        # 输出头：直接输出 2D 噪声预测
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 带噪输入, shape (batch_size, 2)
            t: 时间步, shape (batch_size,)
        
        Returns:
            噪声预测, shape (batch_size, 2)
        """
        features = self.backbone(x, t)
        epsilon_pred = self.output_head(features)
        return epsilon_pred
    
    def get_prediction_components(self, x: torch.Tensor, t: torch.Tensor):
        """
        获取预测的各个分量（用于可视化）
        
        Returns:
            dict: {
                'epsilon': 预测的噪声向量,
                'direction': 归一化后的方向（后处理），
                'norm': 模长（后处理）
            }
        """
        epsilon = self.forward(x, t)
        norm = torch.norm(epsilon, dim=-1, keepdim=True)
        direction = epsilon / (norm + 1e-8)
        
        return {
            'epsilon': epsilon,
            'direction': direction,
            'norm': norm.squeeze(-1)
        }


class NormDecoupledModel(nn.Module):
    """
    模长解耦模型
    分离预测方向（单位向量）和模长（标量）
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 time_dim: int = 64,
                 num_layers: int = 4,
                 rff_scale: float = 30.0):
        super().__init__()
        
        self.backbone = Backbone(input_dim, hidden_dim, time_dim, num_layers, rff_scale=rff_scale)
        
        # 方向头：输出 2D 向量，后接归一化
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 模长头：输出标量，后接 Softplus 保证非负
        self.norm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Softplus 激活函数，保证模长非负
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 带噪输入, shape (batch_size, 2)
            t: 时间步, shape (batch_size,)
        
        Returns:
            重组的噪声预测, shape (batch_size, 2)
        """
        features = self.backbone(x, t)
        
        # 方向预测（归一化为单位向量）
        direction_raw = self.direction_head(features)
        direction = F.normalize(direction_raw, dim=-1)  # ||d|| = 1
        
        # 模长预测（非负）
        norm_raw = self.norm_head(features)
        norm = self.softplus(norm_raw)  # n >= 0
        
        # 重组：ε = n * d
        epsilon = norm * direction
        
        return epsilon
    
    def get_prediction_components(self, x: torch.Tensor, t: torch.Tensor):
        """
        获取预测的各个分量（用于可视化和分析）
        
        Returns:
            dict: {
                'epsilon': 重组后的噪声向量,
                'direction': 预测的方向（单位向量），
                'norm': 预测的模长
            }
        """
        features = self.backbone(x, t)
        
        # 方向预测
        direction_raw = self.direction_head(features)
        direction = F.normalize(direction_raw, dim=-1)
        
        # 模长预测
        norm_raw = self.norm_head(features)
        norm = self.softplus(norm_raw).squeeze(-1)
        
        # 重组
        epsilon = norm.unsqueeze(-1) * direction
        
        return {
            'epsilon': epsilon,
            'direction': direction,
            'norm': norm
        }


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    x = torch.randn(batch_size, 2)
    t = torch.randint(0, 1000, (batch_size,)).float()
    
    print("=" * 50)
    print("Testing Baseline DDPM Model")
    print("=" * 50)
    baseline = BaselineDDPM()
    out_baseline = baseline(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_baseline.shape}")
    print(f"Parameters: {count_parameters(baseline):,}")
    
    components = baseline.get_prediction_components(x, t)
    print(f"Direction shape: {components['direction'].shape}")
    print(f"Norm shape: {components['norm'].shape}")
    
    print("\n" + "=" * 50)
    print("Testing Norm-Decoupled Model")
    print("=" * 50)
    decoupled = NormDecoupledModel()
    out_decoupled = decoupled(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_decoupled.shape}")
    print(f"Parameters: {count_parameters(decoupled):,}")
    
    components = decoupled.get_prediction_components(x, t)
    print(f"Direction shape: {components['direction'].shape}")
    print(f"Direction norm (should be 1): {torch.norm(components['direction'], dim=-1).mean():.6f}")
    print(f"Norm shape: {components['norm'].shape}")
    print(f"Norm values (should be >= 0): min={components['norm'].min():.4f}, max={components['norm'].max():.4f}")
