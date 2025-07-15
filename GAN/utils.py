import os
import numpy as np
from matplotlib import pyplot as plt
from .model import Generator, Discriminator, VAE
from torch.utils.data import Dataset, DataLoader, TensorDataset
from nodivide.utils import speed2traj
import swanlab as wandb
import torch
import torch.nn as nn
from .config import DEVICE, GENERATOR_PATH, GANConfig

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
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.to(DEVICE)

    augmented_imu = augment_data(generator, vel_samples, len(vel_samples))

    # 组合原始和增强数据
    combined_imu = torch.cat([imu_samples, augmented_imu], dim=0)
    combined_vel = torch.cat([vel_samples, vel_samples.clone()], dim=0)

    return combined_imu, combined_vel

def vae_loss_function(x, recon_x, mu, logvar, valid_num):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (valid_num + 1e-8)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum') / (valid_num + 1e-8)
    return KLD + MSE

def draw_vae_samples(model, epoch, dataloader, config, mean, std):
    recon_batch = []
    target_batch = []
    with torch.no_grad():
        for v, m in dataloader:
            v = v.to(DEVICE)
            m = m.to(DEVICE)
            recon, _, _ = model(v)
            m = m.unsqueeze(-1).expand(-1, -1, 2)
            v = v.cpu()
            recon = recon.cpu()
            m = m.cpu()
            target_batch = renorm_vel(v, mean, std, m).numpy()
            recon_batch = renorm_vel(recon, mean, std, m).numpy()
    
    plot_dir = os.path.join(config.vae_dir, config.plots_dir)
    os.makedirs(plot_dir, exist_ok=True)
    for i, (recon, targ) in enumerate(zip(recon_batch, target_batch)):
        plt.figure(figsize=(20, 25))
        plt.subplot(2, 1, 1)
        recon_traj = speed2traj(recon)
        targ_traj = speed2traj(targ)
        plt.plot(recon_traj[:, 0], recon_traj[:, 1], 'r-', label='Predicted', alpha=0.5)
        plt.plot(targ_traj[:, 0], targ_traj[:, 1], 'b-', label='Ground Truth', alpha=0.5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()

        plt.subplot(4, 1, 3)
        time_steps = np.arange(config.seq_len)
        plt.plot(time_steps, recon[:, 0], 'r-', label='Predicted', alpha=0.5)
        plt.plot(time_steps, targ[:, 0], 'b-', label='Ground Truth', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('X')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(time_steps, recon[:, 1], 'r-', label='Predicted', alpha=0.5)
        plt.plot(time_steps, targ[:, 1], 'b-', label='Ground Truth', alpha=0.5)
        plt.xlabel('Time Step') 
        plt.ylabel('Y')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"epoch_{epoch}_sample_{i}.png"))
        wandb.log({f"epoch_{epoch}_sample_{i}": wandb.Image(os.path.join(plot_dir, f"epoch_{epoch}_sample_{i}.png"))})
        plt.close()

def renorm_vel(vel, mean, std, m):
    rn_v = vel * std + mean
    rn_v = rn_v * m
    return rn_v