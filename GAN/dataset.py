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
    def __init__(self, config=VAEConfig):
        self.config = config
        self.seq_length = self.config.seq_len
        self.dataset = IMUTrajectoryDataset()
        all_vel = np.vstack([y for y in self.dataset.y])
        all_vel = all_vel[self.dataset.train_indices]
        self.mean = torch.tensor(np.mean(all_vel, axis=0), dtype=torch.float32)
        self.std = torch.tensor(np.std(all_vel, axis=0), dtype=torch.float32)
        self.velocity_data = []
        self.masks = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        for idx, (y, m, i) in enumerate(zip(self.dataset.y, self.dataset.m, self.dataset.window_idx)):
            norm_y = (y - self.mean) / self.std
            norm_y = norm_y * m.unsqueeze(-1).expand(-1, 2)
            start_idx = 0
            if i != 0:
                start_idx = self.config.full_stride
            for start in range(start_idx, self.config.full_length - self.seq_length + 1, self.config.stride):
                end = start + self.seq_length
                window_y = norm_y[start:end]
                window_m = m[start:end]
                if window_m.sum() <= self.seq_length * 0.4: continue
                if idx in self.dataset.train_indices:
                    self.train_indices.append(len(self.velocity_data))
                elif idx in self.dataset.val_indices:
                    self.val_indices.append(len(self.velocity_data))
                elif idx in self.dataset.test_indices:
                    self.test_indices.append(len(self.velocity_data))
                self.velocity_data.append(window_y)
                self.masks.append(window_m)
        
    def __len__(self):
        return len(self.velocity_data)
    
    def __getitem__(self, idx):
        return self.velocity_data[idx], self.masks[idx]
    
