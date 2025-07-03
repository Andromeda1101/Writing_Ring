import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from .config import *
import numpy as np

def speed2traj(data, fps=200):
    return np.cumsum(data / fps, axis=0)

def speed2point(data, fps=200):
    return torch.cumsum(data / fps, dim=1)

def speed_loss(outputs, targets, masks, alpha=0.8):
    mse = ((outputs - targets) ** 2) * masks
    mse_loss = mse.sum() / (masks.sum() + 1e-8)  
    
    outputs_masked = outputs * masks
    targets_masked = targets * masks
    
    outputs_flat = outputs_masked.view(outputs_masked.size(0), -1)
    targets_flat = targets_masked.view(targets_masked.size(0), -1)
    
    cos_sim = F.cosine_similarity(outputs_flat, targets_flat, dim=1)
    cos_loss = (1 - cos_sim).mean()
    
    return alpha * mse_loss + (1 - alpha) * cos_loss

def traject_loss(outputs, targets, position_weight=0.4, direction_weight=0.3):
    outputs_traj = speed2point(outputs)
    targets_traj = speed2point(targets)
    
    position_loss = torch.mean((outputs_traj - targets_traj) ** 2)
    
    outputs_diff = outputs_traj[:, 1:] - outputs_traj[:, :-1]
    targets_diff = targets_traj[:, 1:] - targets_traj[:, :-1]
    
    outputs_norm = torch.nn.functional.normalize(outputs_diff, dim=-1)
    targets_norm = torch.nn.functional.normalize(targets_diff, dim=-1)
    
    direction_sim = torch.sum(outputs_norm * targets_norm, dim=-1)
    direction_loss = torch.mean(1 - direction_sim)
    
    total_loss = (position_weight * position_loss + 
                 direction_weight * direction_loss)
    
    return total_loss

def draw_trajectory_plots(pred_windows, targ_windows, epoch, window_idx, sample_idx, img_path):
    window_size = TRAIN_CONFIG["time_step"]
    stride = TRAIN_CONFIG["stride"]
    
    plt.figure(figsize=(80, 100))
    plt.subplot(2, 1, 1)
    plt.title(f'Sample {sample_idx} Trajectory (Epoch {epoch}, Window {window_idx})')
    window_pred = pred_windows[window_idx]
    window_targ = targ_windows[window_idx]
    window_pred_traj = speed2traj(window_pred)
    window_targ_traj = speed2traj(window_targ)
    plt.plot(window_pred_traj[:, 0], window_pred_traj[:, 1], 'r-', label='Predicted', alpha=0.5)
    plt.plot(window_targ_traj[:, 0], window_targ_traj[:, 1], 'b-', label='Ground Truth', alpha=0.5)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
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
    plt.savefig(img_path)
    plt.close()
