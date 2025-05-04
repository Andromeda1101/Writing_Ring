# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import DATA_DIR, SAVED_DATA_PATH, TRAIN_CONFIG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, save_processed=True):
        self.data_dir = data_dir
        self.samples = []
        self._load_or_process_data(save_processed)

    def _process_data(self):
        sample_idx = 0  # 
        
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

                            # 物理化
                            y_data[:, 0] = y_data[:, 0] * 24 * 200
                            y_data[:, 1] = y_data[:, 1] * 14 * 200

                            # 归一化
                            x_mean = np.mean(x_data, axis=0)
                            x_std = np.std(x_data, axis=0)
                            x_std[x_std == 0] = 1  # 防止除零
                            x_data = (x_data - x_mean) / x_std
                            
                            self.samples.append({
                                'x': torch.FloatTensor(x_data).contiguous(),
                                'y': torch.FloatTensor(y_data).contiguous(),
                                'm': torch.FloatTensor(m_data).contiguous(),
                                'length': len(x_data),
                                'file_idx': sample_idx,  # 使用全局唯一的索引
                                'file_path': os.path.join(i_path, j_file)  # 保存文件路径以便追踪
                            })
                            sample_idx += 1
        
        print(f'Loaded {len(self.samples)} samples from {self.data_dir}')
        
        total_points = sum(sample['length'] for sample in self.samples)
        print(f'Total data points: {total_points}')
        print(f'Average points per sample: {total_points/len(self.samples):.2f}')

        torch.save({
            'samples': self.samples,
            'total_points': total_points
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
    max_windows = 5000  
    
    sample = batch[0]
    x_data = sample['x']
    y_data = sample['y']
    m_data = sample['m']
    seq_len = sample['length']
    file_idx = sample['file_idx']
    
    # 滑动窗口
    start_positions = list(range(0, seq_len - window_size + 1, stride))
    if not start_positions:
        start_positions = [0]
        current_window_size = seq_len
    else:
        current_window_size = window_size
    
    # 限制窗口数量
    if len(start_positions) > max_windows:
        start_positions = start_positions[:max_windows]


    windows = []
    targets = []
    masks = []
    lengths = []
    indices = []
    
    for start in start_positions:
        end = start + current_window_size
        window = x_data[start:end]
        target = y_data[start:end]
        mask = m_data[start:end]
        
        if len(window) > 0:
            windows.append(window)
            targets.append(target)
            masks.append(mask)
            lengths.append(len(window))
            indices.append(file_idx)
    
    if not windows:  
        raise ValueError("No valid windows found in batch")
    
    # 打包
    windows_padded = pad_sequence(windows, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    masks_padded = pad_sequence(masks, batch_first=True)
    return (
        windows_padded.contiguous(),
        targets_padded.contiguous(),
        masks_padded.contiguous(),
        torch.LongTensor(lengths).contiguous(),
        torch.LongTensor(indices).contiguous()
    )