"""
数据集模块：生成 Swiss Roll 和 2-arm Spiral 数据集
用于验证 Diffusion Model 的模长解耦实验
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class SpiralDataset(Dataset):
    """2-arm Spiral 数据集"""
    
    def __init__(self, n_samples: int = 10000, noise: float = 0.05, seed: int = 42):
        """
        Args:
            n_samples: 样本数量
            noise: 噪声标准差
            seed: 随机种子
        """
        np.random.seed(seed)
        
        # 每臂的样本数
        n_per_arm = n_samples // 2
        
        # 生成螺旋线参数
        # 第一臂
        theta1 = np.sqrt(np.random.rand(n_per_arm)) * 3 * np.pi  # 0 到 3π
        x1 = theta1 * np.cos(theta1)
        y1 = theta1 * np.sin(theta1)
        
        # 第二臂（旋转180度）
        theta2 = np.sqrt(np.random.rand(n_per_arm)) * 3 * np.pi
        x2 = -theta2 * np.cos(theta2)
        y2 = -theta2 * np.sin(theta2)
        
        # 合并两臂
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        
        # 添加噪声
        x += np.random.randn(n_samples) * noise
        y += np.random.randn(n_samples) * noise
        
        # 归一化到 [-1, 1]
        data = np.stack([x, y], axis=1)
        data = self._normalize(data)
        
        self.data = torch.FloatTensor(data)
        
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据到 [-1, 1]"""
        # 找到最大绝对值
        max_abs = np.max(np.abs(data))
        return data / max_abs * 0.9  # 留一点边距
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SwissRollDataset(Dataset):
    """Swiss Roll 数据集（2D投影版本）"""
    
    def __init__(self, n_samples: int = 10000, noise: float = 0.05, seed: int = 42):
        """
        Args:
            n_samples: 样本数量
            noise: 噪声标准差
            seed: 随机种子
        """
        np.random.seed(seed)
        
        # 生成 Swiss Roll 的 t 参数
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
        
        # Swiss Roll 的 x, y 坐标
        x = t * np.cos(t)
        y = t * np.sin(t)
        
        # 添加噪声
        x += np.random.randn(n_samples) * noise
        y += np.random.randn(n_samples) * noise
        
        # 归一化到 [-1, 1]
        data = np.stack([x, y], axis=1)
        data = self._normalize(data)
        
        self.data = torch.FloatTensor(data)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据到 [-1, 1]"""
        max_abs = np.max(np.abs(data))
        return data / max_abs * 0.9
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(dataset_type: str = 'spiral', 
                   n_samples: int = 10000,
                   batch_size: int = 256,
                   noise: float = 0.05,
                   seed: int = 42) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        dataset_type: 'spiral' 或 'swiss_roll'
        n_samples: 样本数量
        batch_size: 批次大小
        noise: 噪声标准差
        seed: 随机种子
    
    Returns:
        DataLoader 对象
    """
    if dataset_type == 'spiral':
        dataset = SpiralDataset(n_samples, noise, seed)
    elif dataset_type == 'swiss_roll':
        dataset = SwissRollDataset(n_samples, noise, seed)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def visualize_dataset(dataset: Dataset, title: str = "Dataset", save_path: str = None):
    """可视化数据集"""
    data = dataset.data.numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, c='blue')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 测试数据集生成
    print("Testing Spiral Dataset...")
    spiral_dataset = SpiralDataset(n_samples=10000)
    print(f"Spiral dataset size: {len(spiral_dataset)}")
    print(f"Data shape: {spiral_dataset.data.shape}")
    print(f"Data range: [{spiral_dataset.data.min():.3f}, {spiral_dataset.data.max():.3f}]")
    visualize_dataset(spiral_dataset, "2-Arm Spiral Dataset", "spiral_dataset.png")
    
    print("\nTesting Swiss Roll Dataset...")
    swiss_dataset = SwissRollDataset(n_samples=10000)
    print(f"Swiss Roll dataset size: {len(swiss_dataset)}")
    visualize_dataset(swiss_dataset, "Swiss Roll Dataset", "swiss_roll_dataset.png")
