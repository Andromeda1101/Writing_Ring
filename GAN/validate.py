import os
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from GAN.config import DEVICE, GANConfig
from GAN.model import VAE, Discriminator, Generator
from GAN.utils import vae_loss_function
from nodivide.utils import speed2traj
import swanlab as wandb

# gan
def gan_validate(generator, discriminator, dataloader, loss_fn, epoch=0):
    if generator is None:
        generator = Generator()
        generator.load_state_dict(torch.load(GANConfig.get_generator_path(GANConfig), map_location=DEVICE))
        generator.to(DEVICE)
    if discriminator is None:
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load(GANConfig.get_discriminator_path(GANConfig), map_location=DEVICE))
        discriminator.to(DEVICE)

    generator.eval()
    discriminator.eval()
    total_g_loss = 0.0
    total_d_loss = 0.0
    plot_num = 0
    samples_fake = []
    samples_targ = []

    with torch.no_grad():
        for batch_idx, (real_imu, real_vel, masks) in tqdm(enumerate(dataloader)):
            batch_size = real_imu.size(0)
            real_imu = real_imu.to(DEVICE)
            real_vel = real_vel.to(DEVICE)
            masks = masks.to(DEVICE)

            # 判别器损失
            valid = torch.ones((batch_size, 1), device=DEVICE)
            fake = torch.zeros((batch_size, 1), device=DEVICE)
            real_loss = loss_fn(discriminator(real_imu, real_vel), valid)

            # 生成器欺骗
            z = torch.randn((batch_size, GANConfig.seq_len, GANConfig.noise_dim), device=DEVICE)
            fake_imu = generator(z, real_vel)
            fake_loss = loss_fn(discriminator(fake_imu.detach(), real_vel), fake)

            d_loss = (real_loss + fake_loss) / 2
            total_d_loss += d_loss.item()

            # 生成器损失
            validity = discriminator(fake_imu, real_vel)
            g_loss = loss_fn(validity, valid)
            total_g_loss += g_loss.item()

            if epoch % GANConfig.plot_freq == 0:
                if plot_num < 5:
                    for fake, targ in zip(fake_imu, real_imu):
                        if plot_num >= 5: break
                        plot_num += 1
                        samples_fake.append(fake.cpu().numpy())
                        samples_targ.append(targ.cpu().numpy())
    if epoch % GANConfig.plot_freq == 0:
        draw_gan_samples(samples_fake, samples_targ, epoch)
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)
    return avg_g_loss, avg_d_loss

def draw_gan_samples(samples_fake, samples_targ, epoch=0):
    os.makedirs(GANConfig.get_plots_dir(GANConfig), exist_ok=True)
    for i, (fake, targ) in tqdm(enumerate(zip(samples_fake, samples_targ))):
        plt.figure(figsize=(20, 25))
        time_steps = np.arange(GANConfig.seq_len)
        plt.subplot(6, 1, 1)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated acc_x', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth acc_x', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('acc_x')
        plt.legend()

        plt.subplot(6, 1, 2)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated acc_y', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth acc_y', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('acc_y')
        plt.legend()

        plt.subplot(6, 1, 3)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated acc_z', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth acc_z', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('acc_z')
        plt.legend()

        plt.subplot(6, 1, 4)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated gyro_x', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth gyro_x', alpha=0.5)
        plt.xlabel('Time Step') 
        plt.ylabel('gyro_x')
        plt.legend()

        plt.subplot(6, 1, 5)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated gyro_y', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth gyro_y', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('gyro_y')
        plt.legend()
        
        plt.subplot(6, 1, 6)
        plt.plot(time_steps, fake[:, i], 'r-', label='Generated gyro_z', alpha=0.5)
        plt.plot(time_steps, targ[:, i], 'b-', label='Ground Truth gyro_z', alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('gyro_z')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(GANConfig.get_plots_dir(GANConfig), f"epoch_{epoch}_sample_{i}.png")
        plt.savefig(plot_path)
        wandb.log({f"epoch_{epoch}_sample_{i}": wandb.Image(plot_path)})
        plt.close()

# vae
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