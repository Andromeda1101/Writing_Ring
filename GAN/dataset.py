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
        return self.seq_length
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
        
class VAEDataset(Dataset):
    def __init__(self):
        self.config = VAEConfig()
        self.seq_length = self.config.seq_length
        data = IMUTrajectoryDataset()
        self.velocity_data = [item['y'] for item in data]
        
    def __len__(self):
        return self.seq_length
    
    def __getitem__(self, idx):
        return self.velocity_data[idx]
    
