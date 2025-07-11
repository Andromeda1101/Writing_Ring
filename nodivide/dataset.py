# dataset.py
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .utils import rotation_perturb
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

def smooth_data(data):
    smooth_data = data.copy()
    slide = 5000
    window_size = 10
    inter_window_size = 4
    is_changed = False
    threshold = 0.5
    max_iter = 10

    for dim in range(2):
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            iter_changed = False
            for start in range(0, len(smooth_data), slide):
                end = min(start + slide, len(smooth_data))
                abs_data = np.abs(smooth_data[start:end, dim])
                max_val = np.max(abs_data)
                if max_val == 0:
                    continue
                max_idx = np.argmax(abs_data)
                def_max_idx = max_idx + start

                start_idx = max(0, def_max_idx - window_size//2)
                inter_start_idx = max(0, def_max_idx - inter_window_size//2)
                inter_end_idx = min(len(smooth_data), def_max_idx + inter_window_size//2 + 1)
                end_idx = min(len(smooth_data), def_max_idx + window_size//2 + 1)
                surroundings_vals_before = np.abs(smooth_data[start_idx:inter_start_idx, dim])
                surroundings_vals_after = np.abs(smooth_data[inter_end_idx:end_idx, dim])
                surroundings_vals = np.concatenate([surroundings_vals_before, surroundings_vals_after])
                max_surrounding_val = np.max(surroundings_vals)

                if max_surrounding_val <= threshold * max_val:
                    is_changed = True
                    iter_changed = True
                    # print(f"Dim {dim}, Iter {iter_count}, Def Max Index: {def_max_idx}, Max Value: {max_val}, Surrounding Max: {max_surrounding_val}")
                    inter_vals = smooth_data[inter_start_idx:inter_end_idx, dim]
                    smooth_data[inter_start_idx:inter_end_idx, dim] = inter_vals * 0.6

            if iter_changed is False:
                break
    
    return smooth_data
    if is_changed is True:
        os.makedirs('smooth_img', exist_ok=True)
        plt.figure(figsize=(15, 15))
        
        # 1
        plt.subplot(2, 2, 1)
        plt.plot(data[:, 0], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_data[:, 0], 'b-', label='Smoothed', alpha=0.5)
        plt.title('D 1')
        plt.legend()
        
        # 2
        plt.subplot(2, 2, 2)
        plt.plot(data[:, 1], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_data[:, 1], 'b-', label='Smoothed', alpha=0.5)
        plt.title('D 2')
        plt.legend()

        # img
        plt.subplot(2, 1, 2)
        orig_points = speed2point(data)
        smooth_points = speed2point(smooth_data)
        plt.plot(orig_points[:, 0], orig_points[:, 1], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-', label='Smoothed', alpha=0.5)
        plt.axis('equal') 
        plt.title('Trajectory Comparison')
        plt.legend()

        plt.tight_layout()
        random_suffix = np.random.randint(1000)
        print(f"Saving smoothed comparison plot with suffix {random_suffix}")
        plt.savefig(f'smooth_img/smooth_comparison_{random_suffix}.png')
        plt.close()

    return smooth_data


