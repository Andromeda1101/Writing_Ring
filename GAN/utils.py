import os
import numpy as np
from matplotlib import pyplot as plt
from .model import Generator, Discriminator, VAE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from nodivide.utils import speed2traj
import swanlab as wandb
import torch
import torch.nn as nn
from .config import DEVICE, GANConfig
from tqdm import tqdm

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
