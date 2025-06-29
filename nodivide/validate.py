import os
from matplotlib.pyplot import plot
import torch
from .dataset import IMUTrajectoryDataset, collate_fn, speed2point
from .model import IMUToTrajectoryNet
from .config import *
import numpy as np
import swanlab as wandb
import matplotlib.pyplot as plt


def validate(model, dataloader, device, epoch=None, plot=True):
    if model is None:
        model = IMUToTrajectoryNet()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(DEVICE)
    model.eval()
    sample_losses = {}
    sample_valid_elements = {}
    samples_pred = {}
    samples_targ = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, lengths, sample_indices, window_indices) in enumerate(dataloader):
            inputs = inputs.contiguous().to(DEVICE)
            targets = targets.contiguous().to(DEVICE)
            masks = masks.contiguous().to(DEVICE)
            lengths = lengths.contiguous().to(DEVICE)
            
            outputs = model(inputs, lengths)
            
            targets = targets[:, :outputs.shape[1], :].contiguous()
            masks = masks[:, :outputs.shape[1]].contiguous()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2).contiguous()
            
            loss_per_element = (outputs - targets) ** 2
            masked_loss = loss_per_element * masks
            
            batch_size = inputs.size(0)
            for i in range(batch_size):
                sample_idx = sample_indices[i].item() 
                
                if sample_idx not in sample_losses:
                    sample_losses[sample_idx] = 0
                    sample_valid_elements[sample_idx] = 0
                
                # 计算当前样本的损失
                sample_loss = masked_loss[i][masks[i] > 0].sum().item()
                valid_elements = masks[i].sum().item()
                
                sample_losses[sample_idx] += sample_loss
                sample_valid_elements[sample_idx] += valid_elements

                # 保存预测和目标轨迹
                if sample_idx not in samples_pred:
                    samples_pred[sample_idx] = {}
                    samples_targ[sample_idx] = {}
                samples_pred[sample_idx][window_indices[i].item()] = outputs[i].cpu().numpy()
                samples_targ[sample_idx][window_indices[i].item()] = targets[i].cpu().numpy()
    
    # 所有样本的平均损失
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else float('inf')
    wandb.log({"val_loss": avg_loss})
    
    if plot and epoch is not None:
        draw_trajectory_plots(samples_pred, samples_targ, epoch)
        for sample_idx in range(len(samples_pred)):
            try:
                img_path = os.path.join('trajectory_plots', f'trajectory_{sample_idx}_epoch_{epoch}.png')
                if os.path.exists(img_path):
                    wandb.log({f"trajectory_plot_{sample_idx}": wandb.Image(img_path)})
            except Exception as e:
                print(f"Error logging trajectory plot: {e}")
    
    return avg_loss

def draw_trajectory_plots(samples_pred, samples_targ, epoch):
    plot_dir = 'trajectory_plots'
    os.makedirs(plot_dir, exist_ok=True)
    window_size = TRAIN_CONFIG["window_size"]
    stride = TRAIN_CONFIG["stride"]

    for sample_idx, pred_windows in samples_pred.items():
        targ_windows = samples_targ[sample_idx]
        
        plt.figure(figsize=(20, 20))

        # 绘制样本轨迹
        plt.subplot(2, 1, 1)
        plt.title(f'Sample {sample_idx} Trajectory (Epoch {epoch})')
        pred_trajectory = []
        targ_trajectory = []
        for window_idx in sorted(pred_windows.keys()):
            if window_idx == 0:
                pred_trajectory.append(pred_windows[window_idx][:window_size-stride, :])
                targ_trajectory.append(targ_windows[window_idx][:window_size-stride, :])
            else:
                pred_trajectory.append(pred_windows[window_idx])
                targ_trajectory.append(targ_windows[window_idx])
            
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-', label='Predicted', alpha=0.5)
        plt.plot(targ_trajectory[:, 0], targ_trajectory[:, 1], 'b-', label='Ground Truth', alpha=0.5)
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.legend()

        # 绘制X坐标随时间的变化
        plt.subplot(2, 2, 3)
        time_steps = np.arange(len(pred_trajectory))
        plt.plot(time_steps, pred_trajectory[:, 0], 'r-', label='Predicted', alpha=0.5)
        plt.plot(time_steps, targ_trajectory[:, 0], 'b-', label='Ground Truth', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('X Position')
        plt.legend()

        # 绘制Y坐标随时间的变化
        plt.subplot(2, 2, 4)
        plt.plot(time_steps, pred_trajectory[:, 1], 'r-', label='Predicted', alpha=0.5)
        plt.plot(time_steps, targ_trajectory[:, 1], 'b-', label='Ground Truth', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('Y Position')
        plt.legend()

        plt.savefig(os.path.join(plot_dir, f'trajectory_sample_{sample_idx}_epoch_{epoch}.png'))
        plt.close()