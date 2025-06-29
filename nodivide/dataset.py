# dataset.py
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
import matplotlib.pyplot as plt
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

                            # 平滑数据
                            y_data = smooth_data(y_data)

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
                                'file_idx': sample_idx,  # 全局唯一索引
                                # 'file_path': os.path.join(i_path, j_file)  # 保存文件路径
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

def collate_fn(batch):
    window_size = TRAIN_CONFIG["time_step"]
    stride = TRAIN_CONFIG["stride"]
    max_windows = 2500  
    
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

def smooth_data(data):
    smooth_data = data.copy()
    window_size = 100  # 窗口大小
    is_changed = False
    threshold = 0.5
    
    for dim in range(2):
        while True:
            abs_data = np.abs(smooth_data[:, dim])
            max_val = np.max(abs_data)
            max_idx = np.argmax(abs_data)

            start_idx = max(0, max_idx - window_size//2)
            end_idx = min(len(abs_data), max_idx + window_size//2 + 1)
            surroundings_vals = abs_data[start_idx:end_idx]
            surroundings_vals[max_idx - start_idx] = 0  
            max_surrounding_val = np.max(surroundings_vals)

            if max_surrounding_val <= threshold * max_val:
                is_changed = True
                smooth_data[max_idx, dim] = max_val * 0.6 * np.sign(smooth_data[max_idx, dim])
            else:
                break

    if is_changed is True:

        os.makedirs('smooth_img', exist_ok=True)
        plt.figure(figsize=(15, 8))
        
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
        plt.subplot(2, 2, 3)
        orig_points = speed2point(data)
        smooth_points = speed2point(smooth_data)
        plt.plot(orig_points[:, 0], orig_points[:, 1], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-', label='Smoothed', alpha=0.5)
        plt.axis('equal') 
        plt.title('Trajectory Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'smooth_img/smooth_comparison_{np.random.randint(1000)}.png')
        plt.close()

    return smooth_data


def speed2point(data, fps=200):
    points = np.zeros_like(data)
    for i in range(1, len(data)):
        points[i] = points[i-1] + data[i] / fps
    return points