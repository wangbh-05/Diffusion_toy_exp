"""
损失函数模块：定义 Baseline MSE Loss 和 Decoupled Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class BaselineLoss:
    """
    标准 DDPM 损失函数
    L = ||ε_gt - ε_pred||²  (MSE Loss)
    """
    
    def __init__(self):
        self.mse = nn.MSELoss()
    
    def __call__(self, 
                 epsilon_pred: torch.Tensor, 
                 epsilon_gt: torch.Tensor,
                 model: Optional[nn.Module] = None,
                 x_t: Optional[torch.Tensor] = None,
                 t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算标准 MSE 损失
        
        Args:
            epsilon_pred: 预测的噪声, shape (batch_size, 2)
            epsilon_gt: 真实噪声, shape (batch_size, 2)
            model: 模型（不使用）
            x_t: 带噪数据（不使用）
            t: 时间步（不使用）
        
        Returns:
            loss: 损失值
            info: 包含详细信息的字典
        """
        loss = self.mse(epsilon_pred, epsilon_gt)
        
        # 计算预测噪声的模长（用于分析）
        pred_norm = torch.norm(epsilon_pred, dim=-1).mean()
        gt_norm = torch.norm(epsilon_gt, dim=-1).mean()
        
        info = {
            'mse_loss': loss.item(),
            'pred_norm': pred_norm.item(),
            'gt_norm': gt_norm.item()
        }
        
        return loss, info


class DecoupledLoss:
    """
    解耦损失函数
    L_total = λ₁ * L_dir + λ₂ * L_norm
    
    其中:
    - L_dir = 1 - CosineSimilarity(ε_gt/||ε_gt||, d_pred)  (方向损失)
    - L_norm = ||ε_gt||₂ - n_pred||²  (模长损失, 可选 Huber Loss)
    """
    
    def __init__(self, 
                 lambda_dir: float = 1.0,
                 lambda_norm: float = 1.0,
                 use_huber: bool = False,
                 huber_delta: float = 1.0,
                 eps: float = 1e-8):
        """
        Args:
            lambda_dir: 方向损失权重
            lambda_norm: 模长损失权重
            use_huber: 是否使用 Huber Loss 计算模长损失
            huber_delta: Huber Loss 的 delta 参数
            eps: 数值稳定性小量
        """
        self.lambda_dir = lambda_dir
        self.lambda_norm = lambda_norm
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.eps = eps
        
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        if use_huber:
            self.norm_loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            self.norm_loss_fn = nn.MSELoss()
    
    def __call__(self,
                 epsilon_pred: torch.Tensor,
                 epsilon_gt: torch.Tensor,
                 model: nn.Module,
                 x_t: torch.Tensor,
                 t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算解耦损失
        
        注意：这个函数需要从模型获取分离的方向和模长预测
        
        Args:
            epsilon_pred: 重组后的预测噪声（暂不使用，直接从模型获取组件）
            epsilon_gt: 真实噪声
            model: NormDecoupledModel，需要有 get_prediction_components 方法
            x_t: 带噪数据
            t: 时间步
        
        Returns:
            loss: 总损失
            info: 详细损失信息
        """
        # 从模型获取分离的预测组件
        components = model.get_prediction_components(x_t, t)
        d_pred = components['direction']  # 预测的方向（单位向量）
        n_pred = components['norm']       # 预测的模长
        
        # 计算真实噪声的方向和模长
        gt_norm = torch.norm(epsilon_gt, dim=-1)  # shape: (batch_size,)
        gt_direction = epsilon_gt / (gt_norm.unsqueeze(-1) + self.eps)  # shape: (batch_size, 2)
        
        # ====== 方向损失 ======
        # 使用余弦相似度损失: L_dir = 1 - cos(gt_direction, d_pred)
        cos_similarity = self.cos_sim(gt_direction, d_pred)  # shape: (batch_size,)
        
        # 处理 gt_norm 接近 0 的情况（此时方向无意义，损失权重降低）
        # 使用软权重来平滑过渡
        direction_weight = torch.clamp(gt_norm / 0.1, 0.0, 1.0)  # gt_norm < 0.1 时降权
        direction_loss = (1.0 - cos_similarity) * direction_weight
        direction_loss = direction_loss.mean()
        
        # ====== 模长损失 ======
        # L_norm = ||gt_norm - n_pred||²
        norm_loss = self.norm_loss_fn(n_pred, gt_norm)
        
        # ====== 总损失 ======
        total_loss = self.lambda_dir * direction_loss + self.lambda_norm * norm_loss
        
        # 记录详细信息
        info = {
            'total_loss': total_loss.item(),
            'direction_loss': direction_loss.item(),
            'norm_loss': norm_loss.item(),
            'cos_similarity': cos_similarity.mean().item(),
            'pred_norm_mean': n_pred.mean().item(),
            'gt_norm_mean': gt_norm.mean().item(),
            'pred_norm_std': n_pred.std().item(),
            'gt_norm_std': gt_norm.std().item()
        }
        
        return total_loss, info


class CombinedDecoupledLoss:
    """
    组合解耦损失：同时包含重组后的 MSE 和分离的方向/模长损失
    L_total = λ_mse * L_mse + λ_dir * L_dir + λ_norm * L_norm
    """
    
    def __init__(self,
                 lambda_mse: float = 1.0,
                 lambda_dir: float = 1.0,
                 lambda_norm: float = 0.5,
                 use_huber: bool = False,
                 eps: float = 1e-8):
        """
        Args:
            lambda_mse: MSE 损失权重
            lambda_dir: 方向损失权重
            lambda_norm: 模长损失权重
            use_huber: 是否使用 Huber Loss
            eps: 数值稳定性小量
        """
        self.lambda_mse = lambda_mse
        self.lambda_dir = lambda_dir
        self.lambda_norm = lambda_norm
        self.eps = eps
        
        self.mse = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.norm_loss_fn = nn.HuberLoss() if use_huber else nn.MSELoss()
    
    def __call__(self,
                 epsilon_pred: torch.Tensor,
                 epsilon_gt: torch.Tensor,
                 model: nn.Module,
                 x_t: torch.Tensor,
                 t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算组合损失"""
        
        # MSE 损失（使用重组后的预测）
        mse_loss = self.mse(epsilon_pred, epsilon_gt)
        
        # 获取分离的组件
        components = model.get_prediction_components(x_t, t)
        d_pred = components['direction']
        n_pred = components['norm']
        
        # 真实噪声的方向和模长
        gt_norm = torch.norm(epsilon_gt, dim=-1)
        gt_direction = epsilon_gt / (gt_norm.unsqueeze(-1) + self.eps)
        
        # 方向损失
        cos_similarity = self.cos_sim(gt_direction, d_pred)
        direction_weight = torch.clamp(gt_norm / 0.1, 0.0, 1.0)
        direction_loss = ((1.0 - cos_similarity) * direction_weight).mean()
        
        # 模长损失
        norm_loss = self.norm_loss_fn(n_pred, gt_norm)
        
        # 总损失
        total_loss = (self.lambda_mse * mse_loss + 
                      self.lambda_dir * direction_loss + 
                      self.lambda_norm * norm_loss)
        
        info = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'direction_loss': direction_loss.item(),
            'norm_loss': norm_loss.item(),
            'cos_similarity': cos_similarity.mean().item(),
            'pred_norm_mean': n_pred.mean().item(),
            'gt_norm_mean': gt_norm.mean().item()
        }
        
        return total_loss, info


def get_loss_function(model_type: str, **kwargs):
    """
    根据模型类型获取对应的损失函数
    
    Args:
        model_type: 'baseline' 或 'decoupled'
        **kwargs: 损失函数的额外参数
    
    Returns:
        损失函数实例
    """
    if model_type == 'baseline':
        return BaselineLoss()
    elif model_type == 'decoupled':
        return DecoupledLoss(**kwargs)
    elif model_type == 'combined':
        return CombinedDecoupledLoss(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from models import BaselineDDPM, NormDecoupledModel
    
    batch_size = 32
    
    # 创建测试数据
    x_t = torch.randn(batch_size, 2)
    t = torch.randint(0, 1000, (batch_size,)).float()
    epsilon_gt = torch.randn(batch_size, 2)
    
    print("=" * 50)
    print("Testing Baseline Loss")
    print("=" * 50)
    
    baseline_model = BaselineDDPM()
    baseline_loss_fn = BaselineLoss()
    
    epsilon_pred = baseline_model(x_t, t)
    loss, info = baseline_loss_fn(epsilon_pred, epsilon_gt)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    print("\n" + "=" * 50)
    print("Testing Decoupled Loss")
    print("=" * 50)
    
    decoupled_model = NormDecoupledModel()
    decoupled_loss_fn = DecoupledLoss(lambda_dir=1.0, lambda_norm=0.5)
    
    epsilon_pred = decoupled_model(x_t, t)
    loss, info = decoupled_loss_fn(epsilon_pred, epsilon_gt, decoupled_model, x_t, t)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 50)
    print("Testing Combined Decoupled Loss")
    print("=" * 50)
    
    combined_loss_fn = CombinedDecoupledLoss(lambda_mse=1.0, lambda_dir=1.0, lambda_norm=0.5)
    loss, info = combined_loss_fn(epsilon_pred, epsilon_gt, decoupled_model, x_t, t)
    print(f"Loss: {loss.item():.4f}")
    print(f"Info:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
