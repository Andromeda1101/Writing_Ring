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
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
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

def vae_validate(model, epoch, dataloader, config, mean, std):
    if model is None:
        model = VAE(config)
        model.load_state_dict(torch.load(os.path.join(config.vae_dir, config.model_path), map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
    model.eval()
    total_losses = []
    plot_num = 0
    samples = []
    with torch.no_grad():
        for v, m in tqdm(dataloader):
            v = v.to(DEVICE)
            m = m.to(DEVICE)
            recon, mu, logvar = model(v)
            m = m.unsqueeze(-1).expand(-1, -1, 2)
            recon = recon * m
            valid_num = m.sum()
            loss = vae_loss_function(v, recon, mu, logvar, valid_num, config.kld_weight)
            total_losses.append(loss.item())

            if epoch % config.test_freq == 0:
                if plot_num < 5:
                    for v, r, m in zip(v, recon, m):
                        if plot_num >= 5: break
                        plot_num += 1
                        samples.append((renorm_vel(v, mean, std, m).cpu().numpy(), renorm_vel(r, mean, std, m).cpu().numpy()))

    avg_loss = np.mean(total_losses)
    if epoch % config.test_freq == 0:
        draw_vae_samples(samples, config, epoch)
    return avg_loss

def draw_vae_samples(samples, config, epoch=0):
    plot_dir = os.path.join(config.vae_dir, config.plots_dir)
    os.makedirs(plot_dir, exist_ok=True)
    for i, (targ, recon) in tqdm(enumerate(samples)):
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