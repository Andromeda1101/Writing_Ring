import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft, stats
from .model import Generator, Discriminator, VAE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from nodivide.utils import speed2traj
import swanlab as wandb
import torch
import torch.nn as nn
from .config import DEVICE, GANConfig
from tqdm import tqdm
from scipy.spatial import ConvexHull

def vel_feat(vel):
    """
    从100×2的速度序列中提取84维区分性特征
    return: 84维特征向量
    """
    if torch.is_tensor(vel):
        vel = vel.cpu().numpy()

    vx = vel[:, 0]
    vy = vel[:, 1]
    vx = np.array(vx, dtype=np.float32)
    vy = np.array(vy, dtype=np.float32)
    v_mag = np.sqrt(vx**2 + vy**2)  # 速度大小
    v_dir = np.arctan2(vy, vx)      # 速度方向
    
    # 1. 基础统计特征 (24维)
    def basic_stats(arr):
        return [
            np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.ptp(arr),
            np.median(arr), np.mean(np.abs(arr)), np.sum(arr**2),
            np.percentile(arr, 25), np.percentile(arr, 75),
            stats.skew(arr), stats.kurtosis(arr)
        ]
    
    features = basic_stats(vx) + basic_stats(vy)
    
    # 2. 频域特征 (20维)
    # 速度大小FFT
    fft_mag = fft.rfft(v_mag)[1:]  # 跳过DC分量
    fft_abs = np.abs(fft_mag)
    features += fft_abs[:5].tolist()  # 前5个幅值
    features += np.angle(fft_mag[:5]).tolist()  # 前5个相位
    features += [np.sum(fft_abs**2) / len(v_mag)]  # 频谱能量
    features += [np.argmax(fft_abs) / len(v_mag)]   # 主频位置
    
    # 方向变化率FFT
    d_theta = np.diff(np.unwrap(v_dir))  # 解卷绕处理角度跳变
    fft_theta = fft.rfft(d_theta)[1:6]  # 取前5个分量
    features += np.abs(fft_theta).tolist()  # 幅值
    features += np.angle(fft_theta).tolist()  # 相位
    
    # 3. 相关性特征 (10维)
    # 速度分量互相关
    for lag in range(-3, 4):
        if lag < 0:
            corr = np.corrcoef(vx[:lag], vy[-lag:])[0, 1]
        else:
            corr = np.corrcoef(vx[lag:], vy[:-lag])[0, 1]
        features.append(corr if not np.isnan(corr) else 0)
    
    # 自相关衰减
    autocorr = np.correlate(vx, vx, mode='full') / np.dot(vx, vx)
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_time = next((i for i, val in enumerate(autocorr) if val < 0.5), 100)
    features.append(autocorr_time)
    
    # 速度-加速度相关性
    ax, ay = np.diff(vx), np.diff(vy)
    corr_va = np.corrcoef(
        np.concatenate([vx[1:], vy[1:]]),
        np.concatenate([ax, ay])
    )[0, 1]
    features.append(corr_va if not np.isnan(corr_va) else 0)
    
    # 4. 运动轨迹特征 (10维)
    # 位移计算
    dx = np.cumsum(vx)
    dy = np.cumsum(vy)
    
    # 轨迹特征
    features.append(np.max(np.sqrt(dx**2 + dy**2)))  # 最大位移
    features.append(np.mean(np.diff(np.arctan2(dy, dx))))  # 平均转向角
    
    # 过零率计算
    def zero_crossing_rate(arr):
        return np.sum(np.diff(np.sign(arr)) != 0) / (2 * len(arr))
    
    features.append(zero_crossing_rate(vx))
    features.append(zero_crossing_rate(vy))
    
    # 加速度过零率
    acc = np.diff(v_mag)
    features.append(zero_crossing_rate(acc))
    
    # 凸包面积
    if len(dx) > 2:
        try:
            hull = ConvexHull(np.column_stack((dx, dy)))
            features.append(hull.volume)
        except:
            features.append(0)
    else:
        features.append(0)
    
    # 分形维度 (简化计算)
    def fractal_dimension(arr):
        n = len(arr)
        scales = np.logspace(0.5, np.log10(n/4), 10, base=10)
        rs = []
        for scale in scales:
            scale = int(scale)
            chunks = n // scale
            r = []
            for i in range(0, n, scale):
                chunk = arr[i:i+scale]
                if len(chunk) > 0:
                    r.append(np.max(chunk) - np.min(chunk))
            rs.append(np.mean(r))
        coeffs = np.polyfit(np.log(scales), np.log(rs), 1)
        return coeffs[0]
    
    features.append(fractal_dimension(dx))
    features.append(fractal_dimension(dy))
    
    # 5. 高阶统计特征 (10维)
    # 多尺度熵 (简化)
    def sample_entropy(arr, scale=1):
        scaled = arr[:len(arr)//scale*scale].reshape(-1, scale).mean(axis=1)
        return np.std(scaled) / np.std(arr)
    
    for scale in [1, 3, 5]:
        features.append(sample_entropy(v_mag, scale))
    
    # Hjorth参数
    diff1 = np.diff(v_mag)
    diff2 = np.diff(diff1)
    activity = np.var(v_mag)
    mobility = np.var(diff1) / activity
    complexity = (np.var(diff2) / np.var(diff1)) / mobility
    features += [activity, mobility, complexity]
    
    # 速度直方图特征
    hist, _ = np.histogram(v_mag, bins=5)
    features += hist.tolist()
    
    # 6. 非线性特征 (10维)
    # Lyapunov指数估计
    def lyapunov_estimate(arr):
        diff = np.abs(np.diff(arr))
        valid = diff > 1e-10
        if np.any(valid):
            return np.mean(np.log(diff[valid]))
        return 0
    
    features.append(lyapunov_estimate(v_mag))
    
    # 递归图特征
    def recurrence_rate(arr, threshold=0.1):
        dist = np.abs(arr[:, None] - arr)
        return np.mean(dist < threshold * np.std(arr))
    
    features.append(recurrence_rate(v_mag))
    
    # 分形维度
    features.append(fractal_dimension(vx))
    features.append(fractal_dimension(vy))
    
    # 速度突变点检测
    def detect_transitions(arr, n=3):
        diff = np.abs(np.diff(arr))
        threshold = np.mean(diff) + 2 * np.std(diff)
        return np.sum(diff > threshold) / len(arr)
    
    features.append(detect_transitions(vx))
    features.append(detect_transitions(vy))
    features.append(detect_transitions(v_mag))
    
    # 多模态检测
    def bimodality_coeff(arr):
        n = len(arr)
        skew = stats.skew(arr)
        kurt = stats.kurtosis(arr)
        return (skew**2 + 1) / (kurt + 3 * (n-1)**2/((n-2)*(n-3)))
    
    features.append(bimodality_coeff(vx))
    features.append(bimodality_coeff(vy))
    
    # 转换为numpy数组
    features = np.array(features)
    
    # 处理可能的NaN值
    features = np.nan_to_num(features)
    
    return torch.FloatTensor(features)

def augment_data(generator, original_vel, num_samples):
    """使用生成器增强数据"""
    generator.eval()
    augmented_imu = []
    
    with torch.no_grad():
        # 分批生成
        for i in range(0, num_samples, GANConfig.batch_size):
            batch_size = min(GANConfig.batch_size, num_samples - i)
            
            # 创建噪声和条件
            z = torch.randn(batch_size, GANConfig.noise_dim, device=DEVICE)
            conditions = original_vel[i:i+batch_size].to(DEVICE)
            
            # 生成IMU数据
            fake_imu = generator(z, conditions)
            augmented_imu.append(fake_imu.cpu())
    
    return torch.cat(augmented_imu, dim=0)

def enhance_dataset_gan(imu_samples, vel_samples):

    generator = Generator()
    generator.load_state_dict(torch.load(GANConfig.get_generator_path(), map_location=DEVICE))
    generator.to(DEVICE)

    augmented_imu = augment_data(generator, vel_samples, len(vel_samples))

    # 组合原始和增强数据
    combined_imu = torch.cat([imu_samples, augmented_imu], dim=0)
    combined_vel = torch.cat([vel_samples, vel_samples.clone()], dim=0)

    return combined_imu, combined_vel

def vae_loss_function(x, recon_x, mu, logvar, valid_num, kld_weight):
    # 添加KLD权重参数
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (valid_num + 1e-8)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum') / (valid_num + 1e-8)
    
    # 使用配置中的KLD权重
    return MSE + kld_weight * KLD
