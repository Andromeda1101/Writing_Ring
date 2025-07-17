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
        self.seq_length = self.config.seq_len
        self.dataset = IMUTrajectoryDataset()
        self.imu_data = []
        self.vel_data = []
        self.masks = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        for idx, (x, y, m, i) in enumerate(zip(self.dataset.x, self.dataset.y, self.dataset.m, self.dataset.window_idx)):
            y = y * m.unsqueeze(-1).expand(-1, 2)
            start_idx = 0
            if i != 0:
                start_idx = self.config.full_stride
            for start in range(start_idx, self.config.full_length - self.seq_length + 1, self.config.stride):
                end = start + self.seq_length
                window_x = x[start:end]
                window_y = y[start:end]
                window_m = m[start:end]
                if window_m.sum().item() <= self.seq_length * 0.5: continue
                if idx in self.dataset.train_indices:
                    self.train_indices.append(len(self.imu_data))
                elif idx in self.dataset.val_indices:
                    self.val_indices.append(len(self.imu_data))
                elif idx in self.dataset.test_indices:
                    self.test_indices.append(len(self.imu_data))
                window_x = torch.FloatTensor(window_x)
                window_y = torch.FloatTensor(window_y)
                window_m = torch.FloatTensor(window_m)
                self.imu_data.append(window_x)
                self.vel_data.append(window_y)
                self.masks.append(window_m)

        self.samples_length = len(self.imu_data)
         
    def __len__(self):
        return self.samples_length
    
    def __getitem__(self, idx):
        return self.imu_data[idx], self.vel_data[idx], self.masks[idx]
        
class VAEDataset(Dataset):
    def __init__(self, config=VAEConfig):
        self.config = config
        self.seq_length = self.config.seq_len
        self.dataset = IMUTrajectoryDataset()
        self.velocity_data = []
        self.masks = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        for idx, (y, m, i) in enumerate(zip(self.dataset.y, self.dataset.m, self.dataset.window_idx)):
            y = y * m.unsqueeze(-1).expand(-1, 2)
            start_idx = 0
            if i != 0:
                start_idx = self.config.full_stride
            for start in range(start_idx, self.config.full_length - self.seq_length + 1, self.config.stride):
                end = start + self.seq_length
                window_y = y[start:end]
                window_m = m[start:end]
                if window_m.sum() <= self.seq_length * 0.2: continue
                if idx in self.dataset.train_indices:
                    self.train_indices.append(len(self.velocity_data))
                elif idx in self.dataset.val_indices:
                    self.val_indices.append(len(self.velocity_data))
                elif idx in self.dataset.test_indices:
                    self.test_indices.append(len(self.velocity_data))
                window_y = torch.FloatTensor(window_y)
                window_m = torch.FloatTensor(window_m)
                self.velocity_data.append(window_y)
                self.masks.append(window_m)

        self.train_indices = np.array(self.train_indices)
        self.val_indices = np.array(self.dataset.val_indices)
        self.test_indices = np.array(self.dataset.test_indices)
        all_vel = np.vstack(self.velocity_data)
        all_vel = torch.FloatTensor(all_vel[self.train_indices])
        self.mean = torch.mean(all_vel, axis=0)
        self.std = torch.std(all_vel, axis=0)
        self.velocity_data = [(v - self.mean) / self.std for v in self.velocity_data]
        
    def __len__(self):
        return len(self.velocity_data)
    
    def __getitem__(self, idx):
        return self.velocity_data[idx], self.masks[idx]
    
