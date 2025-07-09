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
    
    return mse_loss * alpha + (1-alpha) * cos_loss

def traject_loss(outputs, targets, rel_weight=0.8, abs_weight=0.6, dir_weight=0.6):
    outputs_traj = speed2point(outputs)
    targets_traj = speed2point(targets)
    
    # 相对位移损失
    rel_loss = F.mse_loss(
        outputs_traj[:, 1:] - outputs_traj[:, :-1],
        targets_traj[:, 1:] - targets_traj[:, :-1]
    )
    
    # 绝对位置损失
    abs_loss = F.smooth_l1_loss(outputs_traj[:, ::100], targets_traj[:, ::100])

    # 方向损失
    outputs_dir = outputs_traj[:, 1:] - outputs_traj[:, :-1]
    targets_dir = targets_traj[:, 1:] - targets_traj[:, :-1]
    direction_loss = 1 - F.cosine_similarity(outputs_dir, targets_dir, dim=-1).mean()
    
    total_loss = (
        rel_weight * rel_loss 
        + abs_weight * abs_loss
        + dir_weight * direction_loss
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


def rotation_perturb(imu_sample, max_deg=3.0):
    accel_b = imu_sample[..., 0:3].clone()  # [ax, ay, az]
    gyro_b = imu_sample[..., 3:6].clone()   # [gx, gy, gz]
    max_rad = math.radians(max_deg)
    device = imu_sample.device
    
    # 随机旋转轴和角度
    random_axis = torch.randn(3, device=device)
    random_axis = random_axis / torch.linalg.norm(random_axis)  # 归一化
    theta_rad = torch.rand(1, device=device) * max_rad
    
    # 旋转矩阵 R = I + sinθ·K + (1-cosθ)·K²
    K = torch.zeros((3, 3), device=device)
    K[0, 1] = -random_axis[2]
    K[0, 2] = random_axis[1]
    K[1, 0] = random_axis[2]
    K[1, 2] = -random_axis[0]
    K[2, 0] = -random_axis[1]
    K[2, 1] = random_axis[0]
    I = torch.eye(3, device=device)
    R_perturb = I + torch.sin(theta_rad) * K + (1 - torch.cos(theta_rad)) * (K @ K)
    
    accel_b_prime = torch.matmul(accel_b, R_perturb.T)
    gyro_b_prime = torch.matmul(gyro_b, R_perturb.T)
    
    perturbed_sample = torch.cat([accel_b_prime, gyro_b_prime], dim=-1)
    return perturbed_sample