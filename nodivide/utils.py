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

def velocity_loss(outputs, target, masks, alpha=0.8):
    mse_loss = F.mse_loss(outputs, target, reduction='sum') / (masks.sum() + 1e-8)
    
    pred_dir = F.normalize(outputs, dim=-1)
    target_dir = F.normalize(target, dim=-1)
    cos_loss = 1 - F.cosine_similarity(pred_dir, target_dir, dim=-1).mean()
    
    return alpha * mse_loss + (1-alpha) * cos_loss

def traject_loss(outputs, targets, rel_weight=0.8, abs_weight=0.2):
    outputs_traj = speed2point(outputs)
    targets_traj = speed2point(targets)
    
    # 相对位移损失
    rel_loss = F.mse_loss(
        outputs_traj[:, 1:] - outputs_traj[:, :-1],
        targets_traj[:, 1:] - targets_traj[:, :-1]
    )
    
    # 绝对位置损失
    abs_position_def = outputs_traj - targets_traj
    abs_position_def = abs_position_def[:, ::100]
    zero_tensor = torch.zeros_like(abs_position_def)
    abs_loss = F.smooth_l1_loss(abs_position_def, zero_tensor)
    
    total_loss = (
        rel_weight * rel_loss + 
        abs_weight * abs_loss
    )
    
    return total_loss

def draw_trajectory_plots(window_pred, window_targ, epoch, window_idx, sample_idx, img_path):
    window_size = TRAIN_CONFIG["time_step"]
    stride = TRAIN_CONFIG["stride"]
    
    plt.figure(figsize=(80, 100))
    plt.subplot(2, 1, 1)
    plt.title(f'Sample {sample_idx} Trajectory (Epoch {epoch}, Window {window_idx})')

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
