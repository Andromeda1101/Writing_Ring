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

FILE_DIR = "data\\frame_standard_delete_g"

# Cuda Info
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "Not available"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

TIME_STEP = 100  
STRIDE = 20      
INPUT_SIZE = 6   
H_SIZE = 32      
EPOCHS = 100       
NUM_LAYERS = 2   
LR = 0.001      
h_state = None 

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
    def __init__(self, data_dir, window_size=TIME_STEP, stride=STRIDE):
        self.data_dir = data_dir
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        
        sample_idx = 0  # 使用全局计数器
        
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
        # Use GRU
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
            lengths = lengths.cpu()

            packed_input = pack_padded_sequence(
                x, 
                lengths, 
                batch_first=True,
                enforce_sorted=False
            )

            packed_output, _ = self.gru(packed_input)
            
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

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

def train(model, dataloader, optimizer, device):
    model.train()
    sample_losses = {}  
    sample_valid_elements = {}  
    
    total_batches = len(dataloader)
    
    for batch_idx, (inputs, targets, lengths, sample_indices) in enumerate(dataloader):
        if batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        inputs = inputs.contiguous().to(device)
        targets = targets.contiguous().to(device)
        lengths = lengths.contiguous().to(device)
        
        outputs = model(inputs, lengths)
        
        targets = targets[:, :outputs.shape[1], :].contiguous()
        
        loss_per_element = (outputs - targets) ** 2
        
        sample_idx = sample_indices[0].item()
        
        if sample_idx not in sample_losses:
            sample_losses[sample_idx] = 0
            sample_valid_elements[sample_idx] = 0
        
        current_loss = loss_per_element.sum().item()
        current_length = lengths[0].item()
        
        sample_losses[sample_idx] += current_loss
        sample_valid_elements[sample_idx] += current_length * 2
        
        loss = loss_per_element.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record batch loss
        wandb.log({"batch_loss": loss.item()})
    
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else 0
    wandb.log({"train_loss": avg_loss})
    return avg_loss

def validate(model, dataloader, device, epoch=None, plot=False):
    model.eval()
    sample_losses = {}
    sample_valid_elements = {}
    
    # Create plot directory
    if plot:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths, sample_indices) in enumerate(dataloader):
            inputs = inputs.contiguous().to(device)
            targets = targets.contiguous().to(device)
            lengths = lengths.contiguous().to(device)
            
            outputs = model(inputs, lengths)
            targets = targets[:, :outputs.shape[1], :].contiguous()
            
            loss_per_element = (outputs - targets) ** 2
            
            sample_idx = sample_indices[0].item()
            
            if sample_idx not in sample_losses:
                sample_losses[sample_idx] = 0
                sample_valid_elements[sample_idx] = 0
            
            current_loss = loss_per_element.sum().item()
            current_length = lengths[0].item()
            
            sample_losses[sample_idx] += current_loss
            sample_valid_elements[sample_idx] += current_length * 2
            
            # Plot trajectory comparison
            if plot and batch_idx < 5:
                pred_trajectory = outputs[0].cpu().numpy()
                true_trajectory = targets[0].cpu().numpy()
                valid_length = lengths[0].cpu().numpy()
                
                # Get valid trajectory points
                pred_valid = pred_trajectory[:valid_length]
                true_valid = true_trajectory[:valid_length]
                time_valid = np.arange(len(pred_valid))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot X coordinate
                ax1.plot(time_valid, pred_valid[:, 0], 'r-', label='Predicted', linewidth=2)
                ax1.plot(time_valid, true_valid[:, 0], 'b-', label='Ground Truth', linewidth=2)
                ax1.set_title('X Coordinate over Time')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('X Position')
                ax1.legend()
                ax1.grid(True)
                
                # Plot Y coordinate
                ax2.plot(time_valid, pred_valid[:, 1], 'r-', label='Predicted', linewidth=2)
                ax2.plot(time_valid, true_valid[:, 1], 'b-', label='Ground Truth', linewidth=2)
                ax2.set_title('Y Coordinate over Time')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Y Position')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                
                epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
                plt.savefig(os.path.join(plot_dir, f'trajectory_{batch_idx}{epoch_str}.png'))
                plt.close()
    
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else float('inf')
    wandb.log({"val_loss": avg_loss})
    
    # Upload trajectory plots to wandb
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
    # Initialize wandb
    wandb.init(
        project="imu-trajectory-divide",
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

    # CUDA内存配置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)  # 降低内存使用限制
    
    # Check CUDA
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
    
    tscv = TimeSeriesSplit(n_splits=5)  
    
    # 获取最后一次划分的索引
    train_indices = None
    val_indices = None
    for train_idx, val_idx in tscv.split(range(total_size)):
        train_indices = train_idx
        val_indices = val_idx
    
    assert train_indices is not None and val_indices is not None
    
    test_size = int(0.1 * total_size)
    test_start = total_size - test_size
    
    # 不与测试集重叠
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
        batch_size=1,  
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  
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

    # 早停
    best_val_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0
    min_delta = 1e-6  # 最小改善阈值
    
    for epoch in range(EPOCHS):
        print("")
        print("Start Training epoch: ", epoch)
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device, epoch=epoch, plot=True)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_masked_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
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
    
    test_loss = validate(model, test_loader, device, plot=True)
    print(f'Final Test Loss: {test_loss:.8f}')
    wandb.log({"final_test_loss": test_loss})

    # Close wandb
    wandb.finish()
