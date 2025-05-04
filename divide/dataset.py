# dataset.py
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .config import DATA_DIR, SAVED_DATA_PATH, TRAIN_CONFIG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Handle devide
def devide(x_data, y_data, m_data) :
    x_data_d = []
    y_data_d = []
    x_current = []
    y_current = []
    for x, y, m in zip(x_data, y_data, m_data):
        if m != 0 :
            x_current.append(x)
            y_current.append(y)
        else :
            if x_current :
                x_data_d.append(x_current)
                y_data_d.append(y_current)
                x_current = []
                y_current = []

    return x_data_d, y_data_d


class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, save_processed=True):
        self.data_dir = data_dir
        self.samples = []
        self.window_size = TRAIN_CONFIG["time_step"]
        self.stride = TRAIN_CONFIG["stride"]
        self._load_or_process_data(save_processed)

    def _process_data(self):
        sample_idx = 0

        for name in os.listdir(self.data_dir):
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
                        if os.path.exists(y_path) & os.path.exists(m_path):
                            x_data = np.load(os.path.join(i_path, j_file))
                            y_data = np.load(y_path)
                            m_data = np.load(m_path)

                            # 归一化
                            x_mean = np.mean(x_data, axis=0)
                            x_std = np.std(x_data, axis=0)
                            x_std[x_std == 0] = 1  # 防止除零
                            x_data = (x_data - x_mean) / x_std

                            x_data_list, y_data_list = devide(x_data, y_data, m_data)
                            
                            for x, y in zip(x_data_list, y_data_list):
                                x_tensor = torch.from_numpy(np.array(x)).float().contiguous()
                                y_tensor = torch.from_numpy(np.array(y)).float().contiguous()
                                self.samples.append({
                                    'x': x_tensor,
                                    'y': y_tensor,
                                    'length': len(x),
                                    'file_idx': sample_idx,  # 使用全局唯一的索引
                                    'file_path': os.path.join(i_path, j_file)  # 保存文件路径以便追踪
                                })
                                sample_idx += 1

        print(f'Loaded {len(self.samples)} samples from {self.data_dir}')
        total_points = sum(sample['length'] for sample in self.samples)
        print(f'Total data points: {total_points}')
        print(f'Average points per sample: {total_points/len(self.samples):.2f}')
        # 处理完成后保存数据
        torch.save({
            'samples': self.samples,
            'total_points': sum(s['length'] for s in self.samples)
        }, SAVED_DATA_PATH)

    def _load_or_process_data(self, save_processed):
        if os.path.exists(SAVED_DATA_PATH):
            print("Loading preprocessed data...")
            data = torch.load(SAVED_DATA_PATH)
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

def collate_fn(batch):
    window_size = TRAIN_CONFIG["time_step"]
    stride = TRAIN_CONFIG["stride"]
    max_windows = 25000  
    
    sample = batch[0]
    x_data = sample['x']
    y_data = sample['y']
    seq_len = sample['length']
    file_idx = sample['file_idx']
    
    start_positions = list(range(0, seq_len - window_size + 1, stride))
    if not start_positions:
        start_positions = [0]
        current_window_size = seq_len
    else:
        current_window_size = window_size
    
    if len(start_positions) > max_windows:
        start_positions = start_positions[:max_windows]
    
    windows = []
    targets = []
    lengths = []
    indices = []
    
    for start in start_positions:
        end = start + current_window_size
        window = x_data[start:end]
        target = y_data[start:end]
        
        if len(window) > 0:
            windows.append(window)
            targets.append(target)
            lengths.append(len(window))
            indices.append(file_idx)
    
    if not windows: 
        raise ValueError("No valid windows found in batch")
    
    # 打包
    windows_padded = pad_sequence(windows, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    
    return (
        windows_padded.contiguous(),
        targets_padded.contiguous(),
        torch.LongTensor(lengths).contiguous(),
        torch.LongTensor(indices).contiguous()
    )