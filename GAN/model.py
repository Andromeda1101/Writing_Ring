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
            input_size=self.config.noise_dim + self.config.vel_feat_dim,
            hidden_size=512,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout
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
        x = torch.cat((noise, conditions), dim=-1)  # [batch_size, noise_dim + vel_feat_dim]
        gru_out, h_n = self.gru(x)  # output: [batch_size, 512], h_n: [num_layers, batch_size, 512]
        h_n = h_n[-1]  # [batch_size, 512]
        h_n = h_n.view(-1, 512, 1)  # [batch_size, 512, 1] 
        imu_fake = self.deconv(gru_out)  # [batch_size, imu_dim, 16]
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

# VAE 
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        
        # 编码器
        self.encoder_gru = nn.GRU(
            input_size=self.config.input_dim, 
            hidden_size=self.config.hidden_dim, 
            num_layers=self.config.num_layers, 
            dropout=self.config.dropout,
            batch_first=True
        )
        
        self.fc_mu = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        )
        
        # 解码器
        self.decoder_pre = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.input_dim)
        )

        self.decoder_gru = nn.GRU(
            input_size=self.config.latent_dim, 
            hidden_size=self.config.hidden_dim, 
            num_layers=self.config.num_layers, 
            dropout=self.config.dropout,
            batch_first=True
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.input_dim)
        )
        
    def encode(self, x):
        _, h = self.encoder_gru(x)  # h: [num_layers, batch_size, hidden_dim]
        h = h[-1]  # [batch_size, hidden_dim]
        mu = self.fc_mu(h)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(h)  # [batch_size, latent_dim]
        return mu, logvar
    
    def decode(self, z):
        batch_size = z.size(0)  # z: [batch_size, latent_dim]
        z = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        z = z.expand(-1, self.config.seq_len, -1)  # [batch_size, seq_len, latent_dim]
        output, _ = self.decoder_gru(z)  # output: [batch_size, seq_len, hidden_dim]
        recon = self.decoder_fc(output)  # [batch_size, seq_len, input_dim]
        return recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
