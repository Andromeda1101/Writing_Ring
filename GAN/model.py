import torch
import torch.nn as nn
import numpy as np
from .config import *

torch.manual_seed(42)
np.random.seed(42)

config = GANConfig()

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        
        # (noise_dim + vel_dim)
        self.fc = nn.Sequential(
            nn.Linear(config.noise_dim + config.vel_dim, 128),
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
            
            nn.ConvTranspose1d(64, config.imu_dim, kernel_size=4, stride=2, padding=1),
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
        start_idx = (imu_fake.size(2) - config.seq_length) // 2
        return imu_fake[:, :, start_idx:start_idx + config.seq_length]

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        
        self.imu_conv = nn.Sequential(
            # (batch, imu_dim, seq_length)
            nn.Conv1d(config.imu_dim, 64, kernel_size=4, stride=2, padding=1),
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
            nn.Linear(config.vel_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_length = config.seq_length
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