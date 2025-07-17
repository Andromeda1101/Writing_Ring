import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import DEVICE

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embed = np.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=DEVICE) * -embed)
        embed = time[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        return embed

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_mlp(t)[:, :, None]
        h = F.relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim=6, cond_dim=2, time_dim=32, channels=[64, 128, 256]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        self.init_conv = nn.Conv1d(input_dim + cond_dim, channels[0], kernel_size=1)
        
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_dim))
            self.down_pools.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch
        
        self.mid_block = ResidualBlock(channels[-1], channels[-1], time_dim)
        
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            self.up_samples.append(nn.ConvTranspose1d(channels[i], channels[i-1], kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(ResidualBlock(channels[i], channels[i-1], time_dim))
        
        self.final_conv = nn.Conv1d(channels[0], input_dim, kernel_size=1)

    def forward(self, x, cond, t):
        t = self.time_mlp(t)
        
        x = torch.cat([x, cond], dim=1)
        
        x = self.init_conv(x)
        
        skip_connections = []
        for block, pool in zip(self.down_blocks, self.down_pools):
            x = block(x, t)
            skip_connections.append(x)
            x = pool(x)
        
        x = self.mid_block(x, t)
        
        for block, sample, skip in zip(self.up_blocks, self.up_samples, reversed(skip_connections)):
            x = sample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t)
        
        return self.final_conv(x)

