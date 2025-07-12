import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from .config import *
import numpy as np
import types

def smooth_data(data):
    smooth_data = data.copy()
    slide = 5000
    window_size = 10
    inter_window_size = 4
    is_changed = False
    threshold = 0.5
    max_iter = 10

    for dim in range(2):
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            iter_changed = False
            for start in range(0, len(smooth_data), slide):
                end = min(start + slide, len(smooth_data))
                abs_data = np.abs(smooth_data[start:end, dim])
                max_val = np.max(abs_data)
                if max_val == 0:
                    continue
                max_idx = np.argmax(abs_data)
                def_max_idx = max_idx + start

                start_idx = max(0, def_max_idx - window_size//2)
                inter_start_idx = max(0, def_max_idx - inter_window_size//2)
                inter_end_idx = min(len(smooth_data), def_max_idx + inter_window_size//2 + 1)
                end_idx = min(len(smooth_data), def_max_idx + window_size//2 + 1)
                surroundings_vals_before = np.abs(smooth_data[start_idx:inter_start_idx, dim])
                surroundings_vals_after = np.abs(smooth_data[inter_end_idx:end_idx, dim])
                surroundings_vals = np.concatenate([surroundings_vals_before, surroundings_vals_after])
                max_surrounding_val = np.max(surroundings_vals)

                if max_surrounding_val <= threshold * max_val:
                    is_changed = True
                    iter_changed = True
                    # print(f"Dim {dim}, Iter {iter_count}, Def Max Index: {def_max_idx}, Max Value: {max_val}, Surrounding Max: {max_surrounding_val}")
                    inter_vals = smooth_data[inter_start_idx:inter_end_idx, dim]
                    smooth_data[inter_start_idx:inter_end_idx, dim] = inter_vals * 0.6

            if iter_changed is False:
                break
    
    return smooth_data
    if is_changed is True:
        os.makedirs('smooth_img', exist_ok=True)
        plt.figure(figsize=(15, 15))
        
        # 1
        plt.subplot(2, 2, 1)
        plt.plot(data[:, 0], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_data[:, 0], 'b-', label='Smoothed', alpha=0.5)
        plt.title('D 1')
        plt.legend()
        
        # 2
        plt.subplot(2, 2, 2)
        plt.plot(data[:, 1], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_data[:, 1], 'b-', label='Smoothed', alpha=0.5)
        plt.title('D 2')
        plt.legend()

        # img
        plt.subplot(2, 1, 2)
        orig_points = speed2point(data)
        smooth_points = speed2point(smooth_data)
        plt.plot(orig_points[:, 0], orig_points[:, 1], 'r-', label='Original', alpha=0.5)
        plt.plot(smooth_points[:, 0], smooth_points[:, 1], 'b-', label='Smoothed', alpha=0.5)
        plt.axis('equal') 
        plt.title('Trajectory Comparison')
        plt.legend()

        plt.tight_layout()
        random_suffix = np.random.randint(1000)
        print(f"Saving smoothed comparison plot with suffix {random_suffix}")
        plt.savefig(f'smooth_img/smooth_comparison_{random_suffix}.png')
        plt.close()

    return smooth_data



def speed2traj(data, fps=200):
    return np.cumsum(data / fps, axis=0)

def speed2point(data, fps=200):
    return torch.cumsum(data / fps, dim=1)

def velocity_loss(outputs, target, valid_num, alpha=0.8):
    mse_loss = F.mse_loss(outputs, target, reduction='sum') / (valid_num + 1e-8)
    
    pred_dir = F.normalize(outputs, dim=-1)
    target_dir = F.normalize(target, dim=-1)
    cos_loss = 1 - F.cosine_similarity(pred_dir, target_dir, dim=-1).mean()
    
    return mse_loss * alpha + (1-alpha) * cos_loss

def traject_loss(outputs, targets, rel_weight=0.5, abs_weight=0.6, dir_weight=0.3):
    outputs_traj = speed2point(outputs)
    targets_traj = speed2point(targets)
    
    # 相对位移损失
    rel_outputs_traj = torch.diff(outputs_traj, dim=1)
    rel_targets_traj = torch.diff(targets_traj, dim=1)
    rel_loss_x = F.mse_loss(rel_outputs_traj[..., 0], rel_targets_traj[..., 0])
    rel_loss_y = F.mse_loss(rel_outputs_traj[..., 1], rel_targets_traj[..., 1])
    rel_loss = rel_loss_x + rel_loss_y
    # if rel_loss_x > rel_loss_y:
    #     rel_loss = rel_loss_x * 1.5 + rel_loss_y
    # else:
    #     rel_loss = rel_loss_x + rel_loss_y * 1.2
    
    # 绝对位置损失
    abs_loss = F.smooth_l1_loss(outputs_traj, targets_traj)

    # 方向损失
    direction_loss = 1 - F.cosine_similarity(rel_outputs_traj, rel_targets_traj, dim=-1).mean()
    
    total_loss = (
        rel_weight * rel_loss 
        + abs_weight * abs_loss
        + dir_weight * direction_loss
    )
    return total_loss.mean()

def draw_trajectory_plots(window_pred, window_targ, epoch, window_idx, sample_idx, img_path):
    window_size = TRAIN_CONFIG.time_step
    stride = TRAIN_CONFIG.stride
    
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

def rotation_perturb(imu_sample, max_deg=5.0):
    accel_b = imu_sample[..., 0:3].clone()
    gyro_b = imu_sample[..., 3:6].clone()
    max_rad = math.radians(max_deg)
    device = imu_sample.device
    
    # 随机旋转轴和角度
    random_axis = torch.randn(3, device=device)
    random_axis = random_axis / torch.linalg.norm(random_axis)
    theta_rad = torch.rand(1, device=device) * max_rad
    
    K = torch.zeros((3, 3), device=device)
    K[[1, 2, 0], [2, 0, 1]] = random_axis[[2, 0, 1]]  
    K[[2, 0, 1], [1, 2, 0]] = -random_axis[[2, 0, 1]] 
    
    I = torch.eye(3, device=device)
    R_perturb = I + torch.sin(theta_rad) * K + (1 - torch.cos(theta_rad)) * (K @ K)
    
    accel_b_prime = torch.matmul(R_perturb, accel_b.T).T
    gyro_b_prime = torch.matmul(R_perturb, gyro_b.T).T
    
    perturbed_sample = torch.cat([accel_b_prime, gyro_b_prime], dim=-1)
    return perturbed_sample

def class_to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not isinstance(v, types.FunctionType)}