import os
import random
import torch
import torch.nn as nn
from GAN.validate import gan_validate, vae_validate
from nodivide.utils import class_to_dict
from .model import Generator, Discriminator, VAE
from .config import DEVICE, GANConfig, VAEConfig
from .dataset import GANDataset, VAEDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from .utils import vae_loss_function, vel_feat
from tqdm import tqdm
import swanlab as wandb

def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, adversarial_loss):
    generator.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0
    for batch_idx, (real_imu, real_vel, masks) in tqdm(enumerate(dataloader)):
        batch_size = real_imu.size(0)
        
        valid = torch.ones((batch_size, 1), device=DEVICE)
        fake = torch.zeros((batch_size, 1), device=DEVICE)
        
        #  训练判别器
        optimizer_D.zero_grad()
        # 真实样本的损失
        real_imu = real_imu.to(DEVICE)
        real_vel_feat = vel_feat(real_vel).to(DEVICE)
        real_loss = adversarial_loss(discriminator(real_imu, real_vel_feat), valid)
        # 假样本
        z = torch.randn(batch_size, GANConfig.noise_dim, device=DEVICE)
        fake_imu = generator(z, real_vel_feat)
        fake_loss = adversarial_loss(discriminator(fake_imu.detach(), real_vel_feat), fake)
        # 判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        total_d_loss += d_loss.item()
        
        #  训练生成器
        optimizer_G.zero_grad()
        # 生成器欺骗
        validity = discriminator(fake_imu, real_vel_feat)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()
        total_g_loss += g_loss.item()
    
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    return avg_g_loss, avg_d_loss

def train_gan_model():
    torch.manual_seed(42)
    np.random.seed(42)
    config = GANConfig()
    wandb.init(project="ring-gan", config={**class_to_dict(config)})
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # 损失和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr)
    print(f'\nLoading data')
    dataset = GANDataset()
    train_set = Subset(dataset, dataset.train_indices)
    val_set = Subset(dataset, dataset.val_indices)
    test_set = Subset(dataset, dataset.test_indices)
    train_dataloader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f'Training samples: {len(train_set)}')
    print(f'Validation samples: {len(val_set)}')
    print(f'Testing samples: {len(test_set)}')

    for epoch in range(config.epochs):
        
        g_loss, d_loss = train_gan(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, adversarial_loss)
        val_g_loss, val_d_loss = gan_validate(generator, discriminator, val_dataloader, loss_fn=adversarial_loss, epoch=epoch)
        print(f"[Epoch {epoch}/{config.epochs}] [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] [Val D loss: {val_d_loss:.4f}] [Val G loss: {val_g_loss:.4f}]")
        wandb.log({
            "epoch": epoch,
            "generator_loss": g_loss,
            "discriminator_loss": d_loss,
            "val_generator_loss": val_g_loss,
            "val_discriminator_loss": val_d_loss
        })

    # 保存模型
    torch.save(generator.state_dict(), GANConfig.get_generator_path(GANConfig))
    torch.save(discriminator.state_dict(), GANConfig.get_discriminator_path(GANConfig))
    wandb.finish()

def train_vae_model(config=VAEConfig):
    random.seed(42)
    torch.manual_seed(42)
    model = VAE(config=config).to(DEVICE)

    wandb.init(project="ring-vae", config={**class_to_dict(config)})

    
    print(f'\nLoading data')
    full_dataset = VAEDataset(config)
    samples_mean = full_dataset.mean.to(DEVICE)
    samples_std = full_dataset.std.to(DEVICE)

    train_dataset = Subset(full_dataset, full_dataset.train_indices)
    test_dataset = Subset(full_dataset, full_dataset.test_indices)
    val_dataset = Subset(full_dataset, full_dataset.val_indices)
    
    print(f'Total samples: {len(full_dataset)}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Testing samples: {len(test_dataset)}')

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    patience = config.patience
    patience_counter = 0
    min_loss = float('inf')

    for epoch in range(config.epochs):
        total_losses = []
        model.train()
        for batch_idx, (v, m) in tqdm(enumerate(train_loader)):
            v = v.to(DEVICE)
            m = m.to(DEVICE)

            optimizer.zero_grad()
            recon, mu, logvar = model(v)
            m = m.unsqueeze(-1).expand(-1, -1, 2)
            recon = recon * m
            loss = vae_loss_function(v, recon, mu, logvar, valid_num = m.sum(), kld_weight=config.kld_weight)
            total_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # 验证集评估
        val_loss = vae_validate(model, epoch, dataloader=val_loader, config=config, mean=samples_mean, std=samples_std)
        scheduler.step(val_loss)
        avg_loss = np.mean(total_losses)
        print(f"Epoch [{epoch+1}/{config.epochs}] Train Loss: {avg_loss:.4f} Validation Loss: {val_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "train loss":  avg_loss, "val_loss": val_loss})
        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.vae_dir, config.model_path))
            print(f"Model saved at epoch {epoch + 1} with loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                torch.save(model.state_dict(), os.path.join(config.vae_dir, config.final_model_path))
                break
    test_loss = vae_validate(None, 1000, dataloader=test_loader, config=config, mean=samples_mean, std=samples_std)
    print(f'Final Test Loss: {test_loss:.4f}')
    wandb.log({"final_test_loss": test_loss})
    wandb.finish()
        
