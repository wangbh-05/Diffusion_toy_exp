"""
训练模块：完整的训练循环
支持 Baseline 和 Norm-Decoupled 模型的训练
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional, List
import matplotlib.pyplot as plt

from data import get_dataloader, SpiralDataset, SwissRollDataset
from models import BaselineDDPM, NormDecoupledModel
from diffusion import DiffusionScheduler, DDPM, EMA
from losses import get_loss_function


class Trainer:
    """训练器类"""
    
    def __init__(self,
                 model_type: str = 'baseline',
                 dataset_type: str = 'spiral',
                 num_timesteps: int = 1000,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 rff_scale: float = 30.0,
                 lr: float = 1e-3,
                 batch_size: int = 256,
                 n_samples: int = 10000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_ema: bool = True,
                 # Decoupled loss 参数
                 lambda_dir: float = 1.0,
                 lambda_norm: float = 0.5,
                 use_huber: bool = False,
                 exp_name: str = None):
        """
        Args:
            model_type: 'baseline' 或 'decoupled'
            dataset_type: 'spiral' 或 'swiss_roll'
            num_timesteps: 扩散步数
            hidden_dim: 隐藏层维度
            num_layers: 残差块数量
            lr: 学习率
            batch_size: 批次大小
            n_samples: 数据集样本数
            device: 计算设备
            use_ema: 是否使用 EMA
            lambda_dir: 方向损失权重
            lambda_norm: 模长损失权重
            use_huber: 是否使用 Huber Loss
            exp_name: 实验名称
        """
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.device = device
        self.batch_size = batch_size
        self.use_ema = use_ema
        
        # 设置实验名称
        if exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{model_type}_{dataset_type}_{timestamp}"
        self.exp_name = exp_name
        
        # 创建输出目录
        self.output_dir = os.path.join('outputs', exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        
        # 创建数据加载器
        self.dataloader = get_dataloader(
            dataset_type=dataset_type,
            n_samples=n_samples,
            batch_size=batch_size
        )
        
        # 创建模型
        if model_type == 'baseline':
            self.model = BaselineDDPM(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rff_scale=rff_scale
            ).to(device)
        elif model_type == 'decoupled':
            self.model = NormDecoupledModel(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rff_scale=rff_scale
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 创建扩散调度器
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule_type='cosine', # Use cosine schedule for better small data performance
            device=device
        )
        
        # 创建 DDPM 封装
        self.ddpm = DDPM(self.model, self.scheduler, device)
        
        # 创建损失函数
        self.loss_fn = get_loss_function(
            model_type,
            lambda_dir=lambda_dir,
            lambda_norm=lambda_norm,
            use_huber=use_huber
        )
        
        # 创建优化器
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Initialize EMA
        if self.use_ema:
            self.ema = EMA(self.model)
        
        # 记录训练历史
        self.history = {
            'loss': [],
            'metrics': []
        }
        
        # 保存配置
        self.config = {
            'model_type': model_type,
            'dataset_type': dataset_type,
            'num_timesteps': num_timesteps,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'lr': lr,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'lambda_dir': lambda_dir,
            'lambda_norm': lambda_norm,
            'use_huber': use_huber,
            'device': device
        }
        
        # 保存配置到文件
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Trainer initialized:")
        print(f"  Model: {model_type}")
        print(f"  Dataset: {dataset_type}")
        print(f"  Device: {device}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, 
              num_epochs: int = 100,
              log_interval: int = 10,
              sample_interval: int = 20,
              save_interval: int = 50,
              num_samples_viz: int = 1000):
        """
        执行训练
        
        Args:
            num_epochs: 训练轮数
            log_interval: 日志打印间隔
            sample_interval: 采样可视化间隔
            save_interval: 模型保存间隔
            num_samples_viz: 可视化时的采样数量
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # 创建学习率调度器
        lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics = {}
            num_batches = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                x_0 = batch.to(self.device)
                
                # 训练步
                self.optimizer.zero_grad()
                loss, info = self.ddpm.train_step(x_0, self.loss_fn)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                # Update EMA
                if self.use_ema:
                    self.ema.update(self.model)
                
                # 累积统计
                epoch_loss += loss.item()
                for k, v in info.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            # 更新学习率
            lr_scheduler.step()
            
            # 计算平均值
            epoch_loss /= num_batches
            for k in epoch_metrics:
                epoch_metrics[k] /= num_batches
            
            # 记录历史
            self.history['loss'].append(epoch_loss)
            self.history['metrics'].append(epoch_metrics)
            
            # 打印日志
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Loss: {epoch_loss:.6f}")
                print(f"  LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                for k, v in epoch_metrics.items():
                    print(f"  {k}: {v:.6f}")
            
            # 采样可视化
            if (epoch + 1) % sample_interval == 0:
                self._visualize_samples(epoch + 1, num_samples_viz)
            
            # 保存模型
            if (epoch + 1) % save_interval == 0:
                # Apply EMA before saving for checkpointing?
                # Usually we save both, or just EMA.
                # Here we save current model state, and EMA state separately inside checkpoint.
                self._save_checkpoint(epoch + 1)
            
            # 保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_checkpoint('best')
        
        # Apply EMA for final evaluation
        if self.use_ema:
            self.ema.apply_shadow(self.model)
            
        # 保存最终模型
        self._save_checkpoint('final')
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        print(f"\nTraining complete! Best loss: {best_loss:.6f}")
        
        return self.history
    
    def _visualize_samples(self, epoch: int, num_samples: int):
        """生成样本并可视化"""
        # Apply EMA for sampling
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        self.model.eval()
        
        with torch.no_grad():
            samples, trajectory = self.ddpm.sample(
                num_samples,
                return_trajectory=True,
                trajectory_interval=100,
                use_ddim=True # Use DDIM for viz
            )
            
        # Restore original weights for training
        if self.use_ema:
            self.ema.restore(self.model)
        
        samples = samples.cpu().numpy()
        
        # 绘制生成的样本
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f'{self.model_type} - Epoch {epoch}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, 'figures', f'samples_epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_checkpoint(self, tag):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_{tag}.pt')
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(self.history['loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 模长统计（如果有的话）
        if self.history['metrics'] and 'pred_norm_mean' in self.history['metrics'][0]:
            pred_norms = [m.get('pred_norm_mean', 0) for m in self.history['metrics']]
            gt_norms = [m.get('gt_norm_mean', 0) for m in self.history['metrics']]
            
            axes[1].plot(pred_norms, label='Predicted Norm')
            axes[1].plot(gt_norms, label='GT Norm', linestyle='--')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Norm')
            axes[1].set_title('Predicted vs GT Norm')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # 对于 baseline，绘制预测和 GT 的模长
            pred_norms = [m.get('pred_norm', 0) for m in self.history['metrics']]
            gt_norms = [m.get('gt_norm', 0) for m in self.history['metrics']]
            
            axes[1].plot(pred_norms, label='Predicted Norm')
            axes[1].plot(gt_norms, label='GT Norm', linestyle='--')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Norm')
            axes[1].set_title('Predicted vs GT Norm')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'figures', 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {save_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {path}")


def train_both_models(num_epochs: int = 100,
                      dataset_type: str = 'spiral',
                      **kwargs):
    """
    同时训练 Baseline 和 Decoupled 模型，便于对比
    """
    print("=" * 60)
    print("Training Baseline Model")
    print("=" * 60)
    
    baseline_trainer = Trainer(
        model_type='baseline',
        dataset_type=dataset_type,
        exp_name=f'baseline_{dataset_type}',
        **kwargs
    )
    baseline_history = baseline_trainer.train(num_epochs=num_epochs)
    
    print("\n" + "=" * 60)
    print("Training Decoupled Model")
    print("=" * 60)
    
    decoupled_trainer = Trainer(
        model_type='decoupled',
        dataset_type=dataset_type,
        exp_name=f'decoupled_{dataset_type}',
        **kwargs
    )
    decoupled_history = decoupled_trainer.train(num_epochs=num_epochs)
    
    return baseline_trainer, decoupled_trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--model_type', type=str, default='baseline', 
                        choices=['baseline', 'decoupled'])
    parser.add_argument('--dataset', type=str, default='spiral',
                        choices=['spiral', 'swiss_roll'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--lambda_dir', type=float, default=1.0)
    parser.add_argument('--lambda_norm', type=float, default=0.5)
    
    args = parser.parse_args()
    
    trainer = Trainer(
        model_type=args.model_type,
        dataset_type=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        lambda_dir=args.lambda_dir,
        lambda_norm=args.lambda_norm
    )
    
    trainer.train(num_epochs=args.epochs)
