import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .config import *
import numpy as np
from tqdm import tqdm
from nodivide.utils import smooth_data
from nodivide.dataset import IMUTrajectoryDataset

class GANDataset(Dataset):
    def __init__(self):
        self.config = GANConfig()
        self.seq_length = self.config.seq_length
        pre_data = IMUTrajectoryDataset()
        self.data = [(item['x'], item['y']) for item in pre_data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
        
class VAEDataset(Dataset):
    def __init__(self):
        self.config = VAEConfig()
        self.seq_length = self.config.seq_length
        dataset = IMUTrajectoryDataset()
        all_vel = np.vstack([y for y in dataset.y])
        self.mean = torch.tensor(np.mean(all_vel, axis=0), dtype=torch.float32)
        self.std = torch.tensor(np.std(all_vel, axis=0) + 1e-8, dtype=torch.float32)
        # 标准化
        self.velocity_data = []
        for y in dataset.y:
            norm_y = (y - self.mean) / self.std
            self.velocity_data.append(norm_y.reshape(-1))
        
    def __len__(self):
        return len(self.velocity_data)
    
    def __getitem__(self, idx):
        return self.velocity_data[idx]
    
