import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler
from torch import amp

FILE_DIR = "data/frame_standard_delete_g/"  # 修改文件路径只指向cc目录

# 打印系统信息
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "Not available"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

def load_traj(file_path):
    """
    从文件中加载轨迹，轨迹已经归一化到 0-1
    Args:
        file_path: str, 文件路径
    Returns:
        np.ndarray, [n, 2], n 为轨迹点数，第一列为 x 坐标，第二列为 y 坐标
    """
    traj = np.load(file_path)
    # 转换为 cm 单位
    traj[:, 0] = traj[:, 0] * 24
    traj[:, 1] = traj[:, 1] * 13
    return traj

def load_imu_data(file_path):
    """
    从文件中加载 IMU 数据
    Args:
        file_path: str, 文件路径
    Returns:
        np.ndarray, [n, 6], n 为 IMU 数据点数, 6 为 IMU 数据维度
    """
    imu_data = np.load(file_path)
    return imu_data


class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        
        for name in os.listdir(data_dir):
            name_path = os.path.join(data_dir, name)
            for i in os.listdir(name_path):
                i_path = os.path.join(name_path, i)
                if os.path.isdir(i_path):
                    x_files = [f for f in os.listdir(i_path) if f.endswith('_x.npy')]
                    for j_file in x_files:
                        j = j_file.replace('_x.npy', '')
                        y_file = f'{j}_y.npy'
                        if os.path.exists(os.path.join(i_path, y_file)):
                            x_data = np.load(os.path.join(i_path, j_file))
                            y_data = np.load(os.path.join(i_path, y_file))
                            
                            # 过滤掉y值全为0的数据点
                            valid_indices = ~np.all(y_data == 0, axis=1)
                            if np.any(valid_indices):  # 确保至少有一个有效数据点
                                x_data = x_data[valid_indices]
                                y_data = y_data[valid_indices]
                                
                                # 添加数据标准化
                                x_mean = np.mean(x_data, axis=0)
                                x_std = np.std(x_data, axis=0)
                                x_std[x_std == 0] = 1  # 防止除零
                                x_data = (x_data - x_mean) / x_std
                                
                                # 确保数据连续性
                                self.samples.append((
                                    torch.FloatTensor(x_data).contiguous(),
                                    torch.FloatTensor(y_data).contiguous(),
                                    len(x_data)
                                ))
        
        print(f'Loaded {len(self.samples)} samples from {data_dir}')
        
        # 添加统计信息打印
        total_points = sum(sample[2] for sample in self.samples)
        print(f'Total data points: {total_points}')
        print(f'Average points per sample: {total_points/len(self.samples):.2f}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # 分离数据、标签和长度
    x_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    
    # 对序列进行填充，并确保数据连续性
    x_padded = pad_sequence(x_data, batch_first=True).contiguous()
    y_padded = pad_sequence(y_data, batch_first=True).contiguous()
    
    return x_padded, y_padded, torch.LongTensor(lengths)

class IMUToTrajectoryNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, output_size=2, num_layers=3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用单向GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # 改为单向
            dropout=0.1
        )
        
        # 全连接层将GRU的输出映射到轨迹坐标
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 移除 *2 因为不再是双向
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x, lengths):
        # 确保输入数据连续
        x = x.contiguous()
        # Pack padded sequence
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU forward pass
        packed_output, _ = self.gru(packed_x)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply fully connected layer to all time steps
        predictions = self.fc(output)
        
        return predictions

def train_model():
    # 添加CUDA内存配置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存
        # 设置内存分配器配置
        torch.cuda.set_per_process_memory_fraction(0.7)  # 限制GPU内存使用比例
    
    # 检查CUDA
    print(f'CUDA is available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Current CUDA device: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name()}')
        print(f'Device count: {torch.cuda.device_count()}')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据集
    print(f'\nLoading data from: {FILE_DIR}')
    full_dataset = IMUTrajectoryDataset(FILE_DIR)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 训练80%
    test_size = total_size - train_size  # 测试20%
    
    print(f'\nSplitting dataset:')
    print(f'Total samples: {total_size}')
    print(f'Training samples: {train_size}')
    print(f'Testing samples: {test_size}')
    
    # 随机划分数据集
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(31)  
    )
    
    # 创建数据加载器，减小batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 从32减小到16
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16,  # 从32减小到16
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 减小模型规模
    model = IMUToTrajectoryNet(
        input_size=6, 
        hidden_size=256,  # 从512减小到256
        output_size=2,
        num_layers=2      # 从3层减少到2层
    )
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    
    # 调整优化器参数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 调整学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,    # 调整回较温和的学习率衰减
        patience=5,     # 增加耐心值
        verbose=True,
        min_lr=1e-6
    )
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler, device

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x_data, y_data, lengths in dataloader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            lengths = lengths.to(device)
            
            outputs = model(x_data, lengths)
            
            # 计算每个样本的实际损失
            for i in range(len(lengths)):
                seq_len = lengths[i]
                loss = criterion(
                    outputs[i, :seq_len], 
                    y_data[i, :seq_len]
                )
                total_loss += loss.item()
                num_samples += 1
            
            # 及时清理不需要的张量
            del outputs
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    print(f'Validation details:')
    print(f'Total samples: {num_samples}')
    print(f'Total loss: {total_loss:.6f}')
    print(f'Average loss: {avg_loss:.6f}')
    
    model.train()
    return avg_loss
    
def main():
    model, train_loader, test_loader, criterion, optimizer, scheduler, device = train_model()
    
    # 训练轮数
    num_epochs = 10
    best_loss = float('inf')
    
    total_iterations = num_epochs * len(train_loader)
    report_every = max(1, total_iterations // 100)  
    current_iteration = 0
    
    print(f"Total iterations: {total_iterations}")
    print(f"Will report every {report_every} iterations")
    
    # 修正 GradScaler 的创建方式
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for x_data, y_data, lengths in train_loader:
            # 确保数据连续性
            x_data = x_data.contiguous().to(device)
            y_data = y_data.contiguous().to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            # 修正 autocast 的使用方式
            with autocast(enabled=True):
                outputs = model(x_data, lengths)
                mask = torch.arange(y_data.size(1))[None, :].to(device) < lengths[:, None]
                loss = criterion(outputs[mask], y_data[mask])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            batch_count += 1
            current_iteration += 1
            
            # 清理内存
            del outputs, loss, mask
            torch.cuda.empty_cache()
            
            # 每1%输出，监测运行情况
            if current_iteration % report_every == 0:
                avg_loss = running_loss / batch_count
                progress = (current_iteration / total_iterations) * 100
                print(f'Progress: {progress:.1f}% | Epoch: {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # 每个epoch结束后测试
        val_loss = validate_model(model, test_loader, criterion, device)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        
        # 如果学习率发生变化，手动打印信息
        if old_lr != new_lr:
            print(f'\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
        
        # epoch结束时输出
        print(f'\nEpoch {epoch+1}/{num_epochs} completed')
        print(f'Epoch Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Epoch Testing Loss: {val_loss:.4f}')
        print(f'Current Learning Rate: {new_lr:.6f}\n')
        
        if val_loss < best_loss:
            best_loss = val_loss
            print(f'New best model saved! (Test Loss: {val_loss:.4f})')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': running_loss/len(train_loader),
                'test_loss': val_loss,
            }, 'best_model.pth')
    
    # 最终模型
    print("\nFinal Evaluation:")
    model.eval()
    with torch.no_grad():
        test_loss = validate_model(model, test_loader, criterion, device)
        print(f'Final Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()

    