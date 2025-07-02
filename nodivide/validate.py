import os
from matplotlib.pyplot import plot
import torch
from .dataset import speed2point
from .model import IMUToTrajectoryNet
from .config import *
import numpy as np
import swanlab as wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F

def speed_loss(outputs, targets, masks, alpha=0.8):
    mse = ((outputs - targets) ** 2) * masks
    mse_loss = mse.sum() / masks.sum()
    outputs_masked = outputs * masks
    targets_masked = targets * masks
    outputs_flat = outputs_masked.view(outputs_masked.size(0), -1)
    targets_flat = targets_masked.view(targets_masked.size(0), -1)
    cos_sim = F.cosine_similarity(outputs_flat, targets_flat, dim=1)
    cos_loss = 1 - cos_sim.mean()
    return alpha * mse_loss + (1 - alpha) * cos_loss

def traject_loss(outputs, targets):
    outputs_traj = speed2point(outputs.detach().numpy())
    targets_traj = speed2point(targets.detach().numpy())
    traj_dist = outputs_traj - targets_traj
    traj_dist_loss = ((traj_dist[:, 0] ** 2 + traj_dist[:, 1] ** 2)).mean()
    
    outputs_angles = np.arctan2(outputs_traj[:, 1], outputs_traj[:, 0])
    targets_angles = np.arctan2(targets_traj[:, 1], targets_traj[:, 0])
    outputs_grad = np.diff(outputs_angles)
    targets_grad = np.diff(targets_angles)
    outputs_grad = np.where(outputs_grad > np.pi, outputs_grad - 2*np.pi, outputs_grad)
    outputs_grad = np.where(outputs_grad < -np.pi, outputs_grad + 2*np.pi, outputs_grad)
    targets_grad = np.where(targets_grad > np.pi, targets_grad - 2*np.pi, targets_grad)
    targets_grad = np.where(targets_grad < -np.pi, targets_grad + 2*np.pi, targets_grad)
    traj_grad_loss = np.abs(outputs_grad - targets_grad).mean()
    
    return TRAIN_CONFIG["grad_weight"] * traj_grad_loss + TRAIN_CONFIG["dist_weight"] * traj_dist_loss

def validate(model, dataloader, epoch=None, plot=True):
    if model is None:
        model = IMUToTrajectoryNet()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
    model.eval()
    sample_losses = {}
    sample_valid_elements = {}
    samples_pred = {}
    samples_targ = {}
    device = DEVICE
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, sample_indices, window_indices) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.int64)
            outputs = model(inputs, lengths)
            
            outputs = outputs.cpu()
            targets = targets.cpu()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2).cpu()
            
            loss = speed_loss(outputs, targets, masks)
            traj_loss = 0
            
            batch_size = inputs.size(0)
            for i in range(batch_size):
                sample_idx = sample_indices[i].item() 
                
                if sample_idx not in sample_losses:
                    sample_losses[sample_idx] = 0
                    sample_valid_elements[sample_idx] = 0
                
                # 计算当前样本的损失
                sample_loss = ((outputs[i] - targets[i]) ** 2 * masks[i]).sum().item()
                sample_traj_loss = traject_loss(outputs[i] * masks[i], targets[i])
                traj_loss += sample_traj_loss
                valid_elements = masks[i].sum().item()
                
                sample_losses[sample_idx] += sample_loss
                sample_valid_elements[sample_idx] += valid_elements

                # 保存预测和目标轨迹
                if sample_idx not in samples_pred:
                    samples_pred[sample_idx] = {}
                    samples_targ[sample_idx] = {}
                samples_pred[sample_idx][window_indices[i].item()] = (outputs[i] * masks[i]).cpu().numpy()
                samples_targ[sample_idx][window_indices[i].item()] = (targets[i] * masks[i]).cpu().numpy()
    
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
            plt.figure(figsize=(80, 100))
            plt.subplot(2, 1, 1)
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

            plt.subplot(4, 1, 3)
            time_steps = np.arange(window_size)
            plt.plot(time_steps, window_pred[:, 0], 'r-', label='Predicted', alpha=0.5)
            plt.plot(time_steps, window_targ[:, 0], 'b-', label='Ground Truth', alpha=0.5)
            plt.xlabel('Time Step')
            plt.ylabel('X')
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(time_steps, window_pred[:, 1], 'r-', label='Predicted', alpha=0.5)
            plt.plot(time_steps, window_targ[:, 1], 'b-', label='Ground Truth', alpha=0.5)
            plt.xlabel('Time Step') 
            plt.ylabel('Y')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'window_{window_idx}_epoch_{epoch}.png'))
            plt.close()