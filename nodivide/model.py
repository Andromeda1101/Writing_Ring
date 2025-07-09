# model.py
import torch.nn as nn
from .config import MODEL_CONFIG
import torch

class IMUToTrajectoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MODEL_CONFIG


        self.gru = nn.GRU(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=self.config["dropout"]
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(self.config["hidden_size"] * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 卷积层模块
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.config["hidden_size"] * 4, 
                out_channels=self.config["hidden_size"] * 2,  
                kernel_size=5,
                padding=2,
                padding_mode='replicate'
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm([self.config["hidden_size"] * 2, self.config["length"]]),
            
            nn.Conv1d(
                in_channels=self.config["hidden_size"] * 2,
                out_channels=self.config["hidden_size"] * 2,
                kernel_size=3,
                padding=1,
                padding_mode='replicate'
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm([self.config["hidden_size"] * 2, self.config["length"]])
        )
        
        # 全连接层
        self.decoder = nn.Sequential(
            # nn.Linear(self.config["hidden_size"] * 6, 256),
            nn.Linear(self.config["hidden_size"] * 4, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(256, self.config["output_size"]),
        )

    def forward(self, x):
        try:
            # input shape [B, seq_len, 6]
            # GRU
            output, _ = self.gru(x)  # output: [B, seq_len, hidden_size*2]
            
            # 注意力
            attention_weights = self.attention(output)  # [B, seq_len, 1]
            weighted_output = torch.sum(output * attention_weights, dim=1)  # [B, hidden_size*2]
            weighted_output = weighted_output.unsqueeze(1).repeat(1, output.size(1), 1)  # [B, seq_len, hidden_size*2]
            cat_features = torch.cat((output, weighted_output), dim=-1)  # [B, seq_len, hidden_size*4]

            # 卷积
            # conv_input = cat_features.transpose(1, 2)  # [B, hidden_size*4, seq_len]
            # conv_output = self.conv_block(conv_input)  # [B, hidden_size*2, seq_len]
            # 残差连接
            # conv_output = conv_output.transpose(1, 2)  # [B, seq_len, hidden_size*2]
            # conv_features = torch.cat((cat_features, conv_output), dim=-1)  # [B, seq_len, hidden_size*6]

            # 全连接层
            output = self.decoder(cat_features)  # [B, seq_len, output_size]
            # output = self.decoder(conv_features)  # [B, seq_len, output_size]
            
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            raise
