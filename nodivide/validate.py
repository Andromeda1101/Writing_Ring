import os
from matplotlib.pyplot import plot
import torch
from .dataset import IMUTrajectoryDataset, collate_fn
from .model import IMUToTrajectoryNet
from .config import *
import numpy as np
import wandb
import matplotlib.pyplot as plt

def validate(model, dataloader, device, epoch=None, plot=False):
    if model is None:
        model = IMUToTrajectoryNet()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(DEVICE)
    model.eval()
    file_losses = {}
    file_valid_elements = {}
    
    # 创建图表目录
    if plot:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, lengths, file_indices) in enumerate(dataloader):
            inputs = inputs.contiguous().to(DEVICE)
            targets = targets.contiguous().to(DEVICE)
            masks = masks.contiguous().to(DEVICE)
            lengths = lengths.contiguous().to(DEVICE)
            
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