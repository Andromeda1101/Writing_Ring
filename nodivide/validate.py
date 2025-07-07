import os
from matplotlib.pyplot import plot
import torch
import torch.nn.functional as F
from .dataset import IMUTrajectoryDataset, collate_fn, speed2point
from .model import IMUToTrajectoryNet
from .config import *
import numpy as np
import swanlab as wandb
import matplotlib.pyplot as plt


def full_align_loss(outputs, targets, masks, alpha=0.7, beta=0.2, gamma=0.1):
    mse = ((outputs - targets) ** 2) * masks
    mse_loss = mse.sum() / masks.sum()
    # 去均值MSE
    outputs_centered = outputs - (outputs * masks).sum(dim=1, keepdim=True) / (masks.sum(dim=1, keepdim=True) + 1e-8)
    targets_centered = targets - (targets * masks).sum(dim=1, keepdim=True) / (masks.sum(dim=1, keepdim=True) + 1e-8)
    mse_centered = ((outputs_centered - targets_centered) ** 2) * masks
    mse_centered_loss = mse_centered.sum() / masks.sum()
    # 端点对齐损失
    start_mask = masks[:, 0:1, :]
    end_mask = masks[:, -1:, :]
    start_loss = (((outputs[:, 0:1, :] - targets[:, 0:1, :]) ** 2) * start_mask).sum() / (start_mask.sum() + 1e-8)
    end_loss = (((outputs[:, -1:, :] - targets[:, -1:, :]) ** 2) * end_mask).sum() / (end_mask.sum() + 1e-8)
    endpoint_loss = (start_loss + end_loss) / 2
    return alpha * mse_loss + beta * mse_centered_loss + gamma * endpoint_loss

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
            # 损失函数替换为full_align_loss
            loss = full_align_loss(outputs, targets, masks)
            batch_size = inputs.size(0)
            for i in range(batch_size):
                sample_idx = sample_indices[i].item() 
                if sample_idx not in sample_losses:
                    sample_losses[sample_idx] = 0
                    sample_valid_elements[sample_idx] = 0
                sample_loss = ((outputs[i] - targets[i]) ** 2 * masks[i]).sum().item()
                valid_elements = masks[i].sum().item()
                sample_losses[sample_idx] += sample_loss
                sample_valid_elements[sample_idx] += valid_elements
                if sample_idx not in samples_pred:
                    samples_pred[sample_idx] = {}
                    samples_targ[sample_idx] = {}
                samples_pred[sample_idx][window_indices[i].item()] = (outputs[i]*masks[i]).cpu().numpy()
                samples_targ[sample_idx][window_indices[i].item()] = (targets[i]*masks[i]).cpu().numpy()
    # 所有样本的平均损失
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else float('inf')
    wandb.log({"val_loss": avg_loss})
    
    if plot and epoch is not None:
        draw_trajectory_plots(samples_pred, samples_targ, epoch)
        for sample_idx, pred_windows in samples_pred.items():
            if sample_idx // 100 != 9: continue
            window_indices = sorted(pred_windows.keys())
            for window_idx in window_indices:
                try:
                    img_path = os.path.join('trajectory_plots',f'sample_{sample_idx}', f'window_{window_idx}_epoch_{epoch}.png')
                    if os.path.exists(img_path):
                        wandb.log({f"trajectory_plot_{sample_idx}_{window_idx}": wandb.Image(img_path)})
                except Exception as e:
                    print(f"Error logging trajectory plot: {e}")
    
    return avg_loss

def draw_trajectory_plots(samples_pred, samples_targ, epoch):
    plot_dir = 'trajectory_plots'
    os.makedirs(plot_dir, exist_ok=True)
    window_size = TRAIN_CONFIG["time_step"]
    stride = TRAIN_CONFIG["stride"]

    for sample_idx, pred_windows in samples_pred.items():
        if sample_idx // 100 != 9: continue
        sample_dir = os.path.join(plot_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)
        targ_windows = samples_targ[sample_idx]

        window_indices = sorted(pred_windows.keys())
        for window_idx in window_indices:
            plt.figure(figsize=(100, 60))
            plt.subplot(3, 1, 1)
            plt.title(f'Sample {sample_idx} Trajectory (Epoch {epoch}, Window {window_idx})')
            window_pred = pred_windows[window_idx]
            window_targ = targ_windows[window_idx]
            window_pred_traj = speed2point(window_pred)
            window_targ_traj = speed2point(window_targ)
            plt.plot(window_pred_traj[:, 0], window_pred_traj[:, 1], 'r-', label='Predicted', alpha=0.5)
            plt.plot(window_targ_traj[:, 0], window_targ_traj[:, 1], 'b-', label='Ground Truth', alpha=0.5)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.axis('equal')
            plt.legend()

            plt.subplot(3, 1, 2)
            time_steps = np.arange(window_size)
            plt.plot(time_steps, window_pred[:, 0], 'r-', label='Predicted', alpha=0.5)
            plt.plot(time_steps, window_targ[:, 0], 'b-', label='Ground Truth', alpha=0.5)
            plt.xlabel('Time Step')
            plt.ylabel('X')
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(time_steps, window_pred[:, 1], 'r-', label='Predicted', alpha=0.5)
            plt.plot(time_steps, window_targ[:, 1], 'b-', label='Ground Truth', alpha=0.5)
            plt.xlabel('Time Step') 
            plt.ylabel('Y')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'window_{window_idx}_epoch_{epoch}.png'))
            plt.close()