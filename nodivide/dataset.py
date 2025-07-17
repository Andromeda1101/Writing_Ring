# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import smooth_data
from .config import DATA_DIR, SAVED_DATA_PATH, TRAIN_CONFIG
from tqdm import tqdm
import random

class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, save_processed=True):
        self.data_dir = data_dir
        self.samples = []
        self.sample_len = 0
        self.x_mean = 0.0
        self.x_std = 0.0
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.window_size = TRAIN_CONFIG.time_step
        self.stride = TRAIN_CONFIG.stride
        self._load_or_process_data(save_processed)
        

    def _process_data(self):
        sample_idx = 0
        all_x_data = []
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

                            all_x_data.append(x_data)

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
        
        # 划分
        self.sample_len = len(self.samples)
        indices = list(range(self.sample_len))
        random.shuffle(indices)
        
        test_size = int(0.1 * len(indices))
        val_size = int(0.1 * len(indices))
        train_size = len(indices) - test_size - val_size

        self.train_indices = np.array(indices[:train_size])
        self.val_indices = np.array(indices[train_size:train_size + val_size])
        self.test_indices = np.array(indices[train_size + val_size:])

        # 归一化
        train_x = np.vstack(all_x_data)
        train_x = train_x[self.train_indices]
        all_x_tensor = torch.FloatTensor(train_x)
        self.x_mean = torch.mean(all_x_tensor, axis=0)
        self.x_std = torch.std(all_x_tensor, axis=0)
        self.x_std[self.x_std == 0] = 1 

        # 保存处理后的数据
        torch.save({
            'samples': self.samples,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices
        }, SAVED_DATA_PATH)

    def _load_or_process_data(self, save_processed):
        self.x = []
        self.y = []
        self.m = []
        self.sample_idx = []
        self.window_idx = []

        if os.path.exists(SAVED_DATA_PATH):
            print("Loading preprocessed data...")
            data = torch.load(SAVED_DATA_PATH, weights_only=False)
            self.samples = data['samples']
            self.x_mean = data['x_mean']
            self.x_std = data['x_std']
            self.train_indices = data['train_indices']
            self.val_indices = data['val_indices']
            self.test_indices = data['test_indices']

        else:
            print("Processing raw data...")
            self._process_data()
            if save_processed:
                print(f"Saved processed data to {SAVED_DATA_PATH}")
        
        for item in self.samples:
            self.x.append((item['x'] - self.x_mean) / self.x_std) 
            self.y.append(item['y'])
            self.m.append(item['m'])
            self.sample_idx.append(item['sample_idx'])
            self.window_idx.append(item['window_idx'])
        
        self.sample_len = len(self.samples)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx], self.sample_idx[idx], self.window_idx[idx]
    
def get_mean_and_std():
    dataset = IMUTrajectoryDataset()
    return dataset.x_mean, dataset.x_std

