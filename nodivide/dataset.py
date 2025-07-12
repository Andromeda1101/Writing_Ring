# dataset.py
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .utils import rotation_perturb, smooth_data
from .config import DATA_DIR, SAVED_DATA_PATH, TRAIN_CONFIG
from tqdm import tqdm
import random

class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, save_processed=True):
        self.data_dir = data_dir
        self.samples = []
        self.window_size = TRAIN_CONFIG.time_step
        self.stride = TRAIN_CONFIG.stride
        self._load_or_process_data(save_processed)

    def _process_data(self):
        sample_idx = 0
        for name in tqdm(os.listdir(self.data_dir)):
            name_path = os.path.join(self.data_dir, name)
            for i in os.listdir(name_path):
                i_path = os.path.join(name_path, i)
                if os.path.isdir(i_path):
                    x_files = [f for f in os.listdir(i_path) if f.endswith('_x.npy')]
                    for j_file in x_files:
                        j = j_file.replace('_x.npy', '')
                        y_file = f'{j}_y.npy'
                        y_path = os.path.join(i_path, y_file)
                        m_file = f'{j}_mask.npy'
                        m_path = os.path.join(i_path, m_file)
                        if os.path.exists(y_path) and os.path.exists(m_path):
                            x_data = np.load(os.path.join(i_path, j_file))
                            y_data = np.load(y_path)
                            m_data = np.load(m_path)

                            # 物理化
                            y_data[:, 0] = y_data[:, 0] * 24 * 200
                            y_data[:, 1] = y_data[:, 1] * 14 * 200

                            # 平滑数据
                            y_data = smooth_data(y_data)

                            # 归一化
                            # x_mean = np.mean(x_data, axis=0)
                            # x_std = np.std(x_data, axis=0)
                            # x_std[x_std == 0] = 1  # 防止除零
                            # x_data = (x_data - x_mean) / x_std

                            x_tensor = torch.FloatTensor(x_data)
                            y_tensor = torch.FloatTensor(y_data)
                            m_tensor = torch.FloatTensor(m_data)
                            
                            # 划分窗口
                            seq_len = len(x_tensor)
                            for start in range(0, seq_len - self.window_size + 1, self.stride):
                                end = start + self.window_size
                                
                                window_x = x_tensor[start:end]
                                window_y = y_tensor[start:end]
                                window_m = m_tensor[start:end]
                                
                                self.samples.append({
                                    'x': window_x,
                                    'y': window_y,
                                    'm': window_m,
                                    'sample_idx': sample_idx,  # 样本ID
                                    'window_idx': start // self.stride  # 窗口ID
                                })
                            
                            sample_idx += 1
        
        # 打印统计信息
        print(f'Loaded {len(self.samples)} windows from {self.data_dir}')
        print(f'Window size: {self.window_size}, Stride: {self.stride}')
        
        # 保存处理后的数据
        torch.save({
            'samples': self.samples,
            'window_size': self.window_size,
            'stride': self.stride
        }, SAVED_DATA_PATH)

    def _load_or_process_data(self, save_processed):
        if os.path.exists(SAVED_DATA_PATH):
            print("Loading preprocessed data...")
            data = torch.load(SAVED_DATA_PATH, weights_only=True)
            self.samples = data['samples']
        else:
            print("Processing raw data...")
            self._process_data()
            if save_processed:
                print(f"Saved processed data to {SAVED_DATA_PATH}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_collate_fn(batch):
    for item in batch:
        # rand = random.randint(0, 10)
        # if rand <= 3:
        #     item['x'] = rotation_perturb(item['x'])
        # 归一化
        x_mean = torch.mean(item['x'], dim=0)
        x_std = torch.std(item['x'], dim=0)
        x_std[x_std == 0] = 1  # 防止除零
        item['x'] = (item['x'] - x_mean) / x_std

    inputs = torch.stack([item['x'] for item in batch])
    targets = torch.stack([item['y'] for item in batch])
    masks = torch.stack([item['m'] for item in batch])
    sample_idx = torch.tensor([item['sample_idx'] for item in batch])
    window_idx = torch.tensor([item['window_idx'] for item in batch])
    
    return inputs, targets, masks, sample_idx, window_idx

def val_collate_fn(batch):
    for item in batch:
        x_mean = torch.mean(item['x'], dim=0)
        x_std = torch.std(item['x'], dim=0)
        x_std[x_std == 0] = 1  # 防止除零
        item['x'] = (item['x'] - x_mean) / x_std
    inputs = torch.stack([item['x'] for item in batch])
    targets = torch.stack([item['y'] for item in batch])
    masks = torch.stack([item['m'] for item in batch])
    sample_idx = torch.tensor([item['sample_idx'] for item in batch])
    window_idx = torch.tensor([item['window_idx'] for item in batch])
    
    return inputs, targets, masks, sample_idx, window_idx


