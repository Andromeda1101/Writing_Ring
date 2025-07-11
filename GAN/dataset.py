from torch.utils.data import Dataset, DataLoader, TensorDataset
from .config import *
import numpy as np

class IMUDataset(Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
            self.config = GANConfig()
            self.seq_length = self.config.seq_length
            
            # 生成模拟数据 (实际应用中应替换为真实数据)
            self.imu_data = np.random.randn(num_samples, self.config.imu_dim, self.seq_length).astype(np.float32)
            self.velocity_data = np.random.randn(num_samples, self.config.vel_dim).astype(np.float32)
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.imu_data[idx], self.velocity_data[idx]