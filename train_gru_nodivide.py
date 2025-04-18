import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import amp
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import wandb

# torch.backends.cudnn.enabled = False  # 禁用cuDNN

FILE_DIR = "data/frame_standard_delete_g" 

# Check Cuda
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "Not available"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

TIME_STEP = 100  
STRIDE = 60      
INPUT_SIZE = 6   
H_SIZE = 64      
EPOCHS = 200       
NUM_LAYERS = 2   
LR = 0.001      
h_state = None 

class IMUTrajectoryDataset(Dataset):
    def __init__(self, data_dir, window_size=TIME_STEP, stride=STRIDE):
        self.data_dir = data_dir
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        
        sample_idx = 0  # 
        
        for name in os.listdir(data_dir):
            name_path = os.path.join(data_dir, name)
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
        
        print(f'Loaded {len(self.samples)} samples from {data_dir}')
        
        total_points = sum(sample['length'] for sample in self.samples)
        print(f'Total data points: {total_points}')
        print(f'Average points per sample: {total_points/len(self.samples):.2f}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class IMUToTrajectoryNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=1, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, lengths):
        try:
            assert not torch.isnan(x).any(), "NaN in input"
            assert (lengths > 0).all(), "Non-positive lengths"
            
            x = x.contiguous()
            lengths = lengths.cpu().contiguous()
            
            # 打包
            packed_input = pack_padded_sequence(
                x, 
                lengths, 
                batch_first=True, 
                enforce_sorted=True
            )
            
            # GRU处理
            packed_output, _ = self.gru(packed_input)
            
            # 解包
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            output = output.contiguous()
            
            # 使用全连接层
            batch_size, seq_len, hidden_size = output.size()
            output = output.view(-1, hidden_size)
            output = self.fc(output)
            output = output.view(batch_size, seq_len, -1)
            
            return output.contiguous()
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            print(f"Lengths: {lengths}")
            raise

def collate_fn(batch):
    window_size = TIME_STEP
    stride = STRIDE
    max_windows = 1000  
    
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

def train(model, dataloader, optimizer, device):
    model.train()
    sample_losses = {}  
    sample_valid_elements = {}  
    
    total_batches = len(dataloader)
    # progress_interval = total_batches // 3
    
    for batch_idx, (inputs, targets, masks, lengths, sample_indices) in enumerate(dataloader):
        if batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        inputs = inputs.contiguous().to(device)
        targets = targets.contiguous().to(device)
        masks = masks.contiguous().to(device)
        lengths = lengths.contiguous().to(device)
        
        outputs = model(inputs, lengths)
        
        # 维度匹配
        targets = targets[:, :outputs.shape[1], :].contiguous()
        masks = masks[:, :outputs.shape[1]].contiguous()
        masks = masks.unsqueeze(-1).expand(-1, -1, 2).contiguous()
        
        # 计算每个窗口的损失
        loss_per_element = (outputs - targets) ** 2
        # 只计算mask为1的部分的loss
        valid_loss = loss_per_element[masks > 0]
        masked_loss = valid_loss if len(valid_loss) > 0 else torch.tensor([0.0]).to(device)
        
        # 获取索引
        sample_idx = sample_indices[0].item()
        
        # 累积损失
        if sample_idx not in sample_losses:
            sample_losses[sample_idx] = 0
            sample_valid_elements[sample_idx] = 0
        
        current_loss = masked_loss.sum().item()
        current_valid = masks.sum().item()
        
        sample_losses[sample_idx] += current_loss
        sample_valid_elements[sample_idx] += current_valid
        
        # 计算当前批次的平均损失用于反向传播
        current_valid = masks.sum()
        if current_valid > 0:
            loss = masked_loss.sum() / current_valid
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录每个batch的损失
            wandb.log({"batch_loss": loss.item()})
        
        # 显示进度
        # if (batch_idx + 1) % progress_interval == 0:
        #     progress = (batch_idx + 1) / total_batches * 100
        #     avg_loss = sum(sample_losses.values()) / sum(sample_valid_elements.values()) if sample_valid_elements else 0
        #     print(f'Training Progress: {progress:.0f}% - Current Loss: {avg_loss:.8f}')
    
    # 返回所有样本的平均损失
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else 0
    wandb.log({"train_loss": avg_loss})
    return avg_loss

def validate(model, dataloader, device, epoch=None, plot=False):
    model.eval()
    file_losses = {}
    file_valid_elements = {}
    
    # 创建图表目录
    if plot:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, lengths, file_indices) in enumerate(dataloader):
            inputs = inputs.contiguous().to(device)
            targets = targets.contiguous().to(device)
            masks = masks.contiguous().to(device)
            lengths = lengths.contiguous().to(device)
            
            outputs = model(inputs, lengths)
            
            # 确保维度匹配
            targets = targets[:, :outputs.shape[1], :].contiguous()
            masks = masks[:, :outputs.shape[1]].contiguous()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2).contiguous()
            
            # 计算每个窗口的损失
            loss_per_element = (outputs - targets) ** 2
            masked_loss = loss_per_element * masks
            
            file_idx = file_indices[0].item()
            if file_idx not in file_losses:
                file_losses[file_idx] = 0
                file_valid_elements[file_idx] = 0
            
            current_loss = masked_loss.sum().item()
            current_valid = masks.sum().item()
            
            file_losses[file_idx] += current_loss
            file_valid_elements[file_idx] += current_valid

            # 绘制轨迹对比图
            if plot and batch_idx < 5:  # 只绘制前5个样本
                # 将数据移到CPU并转换为numpy数组
                pred_trajectory = outputs[0].cpu().numpy()
                true_trajectory = targets[0].cpu().numpy()
                mask = masks[0, :, 0].cpu().numpy()  # 只需要一个通道的mask

                # 获取有效的轨迹点（根据mask）
                valid_indices = mask > 0
                pred_valid = pred_trajectory[valid_indices]
                true_valid = true_trajectory[valid_indices]
                
                # 创建时间数组
                time_valid = np.arange(len(pred_valid))

                # 创建两个子图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # 绘制X坐标随时间的变化
                ax1.plot(time_valid, pred_valid[:, 0], 'r-', label='Predicted', linewidth=2)
                ax1.plot(time_valid, true_valid[:, 0], 'b-', label='Ground Truth', linewidth=2)
                ax1.set_title('X Coordinate over Time')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('X Position')
                ax1.legend()
                ax1.grid(True)
                
                # 绘制Y坐标随时间的变化
                ax2.plot(time_valid, pred_valid[:, 1], 'r-', label='Predicted', linewidth=2)
                ax2.plot(time_valid, true_valid[:, 1], 'b-', label='Ground Truth', linewidth=2)
                ax2.set_title('Y Coordinate over Time')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Y Position')
                ax2.legend()
                ax2.grid(True)
                
                # 调整子图之间的间距
                plt.tight_layout()
                
                # 保存图片
                epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
                plt.savefig(os.path.join(plot_dir, f'trajectory_{batch_idx}{epoch_str}.png'))
                plt.close()
    
    # 返回所有文件的平均损失
    total_loss = sum(file_losses.values())
    total_valid = sum(file_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else float('inf')
    wandb.log({"val_loss": avg_loss})
    
    # 如果绘制了轨迹图，上传到wandb
    if plot and epoch is not None:
        for batch_idx in range(min(5, len(dataloader))):
            try:
                img_path = os.path.join('trajectory_plots', f'trajectory_{batch_idx}_epoch_{epoch}.png')
                if os.path.exists(img_path):
                    wandb.log({f"trajectory_plot_{batch_idx}": wandb.Image(img_path)})
            except Exception as e:
                print(f"Error logging trajectory plot: {e}")
    
    return avg_loss

if __name__ == "__main__":
    # 初始化wandb
    wandb.init(
        project="imu-trajectory",
        config={
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": 1,
            "hidden_size": H_SIZE,
            "num_layers": NUM_LAYERS,
            "time_step": TIME_STEP,
            "stride": STRIDE
        }
    )
    
    # 添加CUDA内存配置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)  # 降低内存使用限制
    
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
    
    # 使用TimeSeriesSplit进行数据集划分
    tscv = TimeSeriesSplit(n_splits=5)  # 5折交叉验证
    
    # 获取最后一次划分的索引
    train_indices = None
    val_indices = None
    for train_idx, val_idx in tscv.split(range(total_size)):
        train_indices = train_idx
        val_indices = val_idx
    
    # 确保我们有训练集和验证集
    assert train_indices is not None and val_indices is not None
    
    # 计算测试集大小（使用最后10%的数据）
    test_size = int(0.1 * total_size)
    test_start = total_size - test_size
    
    # 调整验证集，不要与测试集重叠
    val_indices = val_indices[val_indices < test_start]
    
    # 创建数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, range(test_start, total_size))
    
    print(f'\nSplitting dataset:')
    print(f'Total samples: {total_size}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Testing samples: {len(test_dataset)}')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # 每次只处理一个样本
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 每次只处理一个样本
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # 每次只处理一个样本
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    model = IMUToTrajectoryNet(
        input_size=6,
        hidden_size=H_SIZE,
        num_layers=NUM_LAYERS,
        output_size=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=LR,
                                weight_decay=1e-5)

    # 修改早停策略
    best_val_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0
    min_delta = 1e-6  # 最小改善阈值
    
    for epoch in range(EPOCHS):
        print("")
        print("Start Training epoch: ", epoch)
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device, epoch=epoch, plot=True)

        # 检查是否有显著改善
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_masked_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 学习率调整和早停
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        print(f'Epoch [{epoch+1}/{EPOCHS}], '
              f'Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "patience_counter": patience_counter
        })
    
    # model.load_state_dict(torch.load('best_masked_model.pth'))
    test_loss = validate(model, test_loader, device, plot=True)
    print(f'Final Test Loss: {test_loss:.8f}')
    wandb.log({"final_test_loss": test_loss})

    # 在程序结束时关闭wandb
    wandb.finish()
