import os
from .model import *
from .config import *
from .dataset import IMUDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss):
    total_g_loss = 0.0
    total_d_loss = 0.0
    for batch_idx, (real_imu, real_vel) in enumerate(dataloader):
        batch_size = real_imu.size(0)
        
        valid = torch.ones((batch_size, 1), device=DEVICE)
        fake = torch.zeros((batch_size, 1), device=DEVICE)
        
        #  训练判别器
        optimizer_D.zero_grad()
        # 真实样本的损失
        real_imu = real_imu.to(DEVICE)
        real_vel = real_vel.to(DEVICE)
        real_loss = adversarial_loss(discriminator(real_imu, real_vel), valid)
        # 假样本
        z = torch.randn(batch_size, config.noise_dim, device=DEVICE)
        fake_imu = generator(z, real_vel)
        fake_loss = adversarial_loss(discriminator(fake_imu.detach(), real_vel), fake)
        # 判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        total_d_loss += d_loss
        
        #  训练生成器
        optimizer_G.zero_grad()
        # 生成器欺骗
        validity = discriminator(fake_imu, real_vel)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss
    
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

def train_model():
    generator = Generator(config).to(DEVICE)
    discriminator = Discriminator(config).to(DEVICE)

    # 损失和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr)
    dataset = IMUDataset(num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        g_loss, d_loss = train(generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss)
            
        print(f"[Epoch {epoch}/{config.epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        if epoch % config.sample_interval == 0:
            z = torch.randn(5, config.noise_dim, device=DEVICE)
            sample_vel = torch.randn(5, config.vel_dim, device=DEVICE)
            gen_samples = generator(z, sample_vel).detach().cpu().numpy()
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 10))
            axes = axes.flatten()
            for j in range(6):  # 6个IMU通道
                ax = axes[j]
                for k in range(5):  # 5个样本
                    ax.plot(gen_samples[k, j], alpha=0.7)
                ax.set_title(f'IMU Channel {j+1}')
                ax.grid(True)
            plt.suptitle(f'Epoch {epoch} - Generated IMU Samples')
            plt.tight_layout()
            path = os.path.join(SAMPLES_PATH, f"generated_samples_epoch_{epoch}.png")
            plt.savefig(path)
            plt.close()

    # 保存模型
    torch.save(generator.state_dict(), GENERATOR_PATH)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_PATH)