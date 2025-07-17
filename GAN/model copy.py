import torch
import torch.nn as nn
import numpy as np
from .config import GANConfig
# GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.config = GANConfig()
        
        self.gru = nn.GRU(
            input_size=self.config.noise_dim + self.config.vel_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.deconv = nn.Sequential(
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
        
        self.upsample = nn.Upsample(size=self.config.seq_len, mode='linear', align_corners=False)
        
    def forward(self, noise, conditions):
        x = torch.cat((noise, conditions), dim=-1)  # [batch_size, seq_len, noise_dim + vel_dim]
        _, h_n = self.gru(x)  # h_n: [num_layers, batch_size, 512]
        x = h_n[-1]  # [batch_size, 512]
        x = x.view(-1, 512, 1)  # [batch_size, 512, 1]
        imu_fake = self.deconv(x)  # [batch_size, imu_dim, 16]
        imu_fake = self.upsample(imu_fake)  # [batch_size, imu_dim, seq_len]
        imu_fake = imu_fake.permute(0, 2, 1)  # [batch_size, seq_len, imu_dim]
        return imu_fake

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.config = GANConfig()
        
        self.imu_conv = nn.Sequential(
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
        
        with torch.no_grad():
            test_input = torch.zeros(1, self.config.imu_dim, self.config.seq_len)
            test_output = self.imu_conv(test_input)
            self.conv_length = test_output.size(2)
            # print(f"Conv output length: {self.conv_length}")
        
        self.condition_fc = nn.Sequential(
            nn.Linear(self.config.vel_dim * self.config.seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * self.conv_length + 512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, imu, conditions):
        imu = imu.permute(0, 2, 1)  # [batch_size, imu_dim, seq_len]
        imu_features = self.imu_conv(imu)  # [batch_size, 512, conv_length]
        conditions = conditions.view(conditions.size(0), -1)  # [batch_size, vel_dim * seq_len]
        condition_features = self.condition_fc(conditions)  # [batch_size, 512]

        imu_flat = imu_features.view(imu_features.size(0), -1)  # [batch_size, 512 * conv_length]
        condition_features = condition_features.view(condition_features.size(0), -1)  # [batch_size, 512]
        combined = torch.cat((imu_flat, condition_features), dim=1)  # [batch_size, 512 * conv_length + 512]
        validity = self.fc(combined)  # [batch_size, 1]
        return validity

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        out = self.upconv(x)
        out = torch.cat([out, skip], dim=1)
        return out

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        
        # Encoder (U-Net下采样路径)
        self.enc1 = UNetBlock(self.config.input_dim, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        # Latent space
        self.fc_mu = nn.Linear(512 * (self.config.seq_len // 16), self.config.latent_dim)
        self.fc_logvar = nn.Linear(512 * (self.config.seq_len // 16), self.config.latent_dim)
        
        # Decoder (U-Net上采样路径)
        latent_seq_len = self.config.seq_len // 16
        self.latent_fc = nn.Linear(self.config.latent_dim, 512 * latent_seq_len)
        
        self.dec4 = UNetUpBlock(512, 256)
        self.dec3 = UNetUpBlock(512, 128)
        self.dec2 = UNetUpBlock(256, 64)
        self.dec1 = UNetUpBlock(128, 32)
        
        self.final_conv = nn.Conv1d(64, self.config.input_dim, 1)
        
    def encode(self, x):
        # Convert [batch, seq_len, channels] to [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # Encoder path with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Flatten and project to latent space
        flat = e4.view(e4.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        
        return mu, logvar, (e1, e2, e3, e4)
    
    def decode(self, z, skip_connections):
        e1, e2, e3, e4 = skip_connections
        
        # Reshape latent vector
        z = self.latent_fc(z)
        z = z.view(z.size(0), 512, -1)
        
        # Decoder path using skip connections
        d4 = self.dec4(z, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        out = self.final_conv(d1)
        # Convert back to [batch, seq_len, channels]
        out = out.permute(0, 2, 1)
        
        return out
    
    def forward(self, x):
        mu, logvar, skip_connections = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, skip_connections)
        return recon_x, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
