import torch
import torch.nn as nn
import numpy as np
from .config import *

# GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.config = GANConfig()
        
        # (noise_dim + vel_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.config.noise_dim + self.config.vel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            # 输入: (batch, 512, 1)
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, self.config.imu_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # 计算输出长度
        self.output_length = 1
        for _ in range(4): 
            self.output_length = (self.output_length - 1) * 2 + 4
        
    def forward(self, noise, conditions):
        # 拼接噪声和条件向量
        x = torch.cat((noise, conditions), dim=1)
        
        # 全连接层
        x = self.fc(x)
        x = x.view(-1, 512, 1)
        
        # 转置卷积
        imu_fake = self.deconv(x)
        
        # 裁剪
        start_idx = (imu_fake.size(2) - self.config.seq_length) // 2
        return imu_fake[:, :, start_idx:start_idx + self.config.seq_length]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.config = GANConfig()
        
        self.imu_conv = nn.Sequential(
            # (batch, imu_dim, seq_length)
            nn.Conv1d(self.config.imu_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.condition_fc = nn.Sequential(
            nn.Linear(self.config.vel_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_length = self.config.seq_length
        for _ in range(4): 
            self.conv_length = (self.conv_length + 2 - 4) // 2 + 1
        
        # 判别层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.conv_length + 512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, imu, conditions):
        imu_features = self.imu_conv(imu)
        condition_features = self.condition_fc(conditions)
        
        imu_flat = imu_features.view(imu_features.size(0), -1)
        combined = torch.cat((imu_flat, condition_features), dim=1)
        
        validity = self.fc(combined)
        return validity

# VAE 
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.config = VAEConfig()
        self.encoder = Encoder(self.config.input_dim, self.config.hidden_dim, self.config.latent_dim)
        self.decoder = Decoder(self.config.input_dim, self.config.hidden_dim, self.config.latent_dim)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # 计算标准差, std = sqrt(var) = sqrt(exp(logvar)) = exp(logvar/2)
        epsilon = torch.randn_like(std, requires_grad=False) # 从标准正态分布中采样epsilon
        return mu + epsilon * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
