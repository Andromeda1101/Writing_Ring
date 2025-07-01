# model.py
import torch.nn as nn
from .config import MODEL_CONFIG
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IMUToTrajectoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MODEL_CONFIG

        self.conv_pre = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
        
        self.gru = nn.GRU(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            batch_first=True,
            dropout=self.config["dropout"]
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(self.config["hidden_size"], 1),
            nn.Softmax(dim=1)
        )
        
        # 全连接层
        self.fc = nn.Linear(self.config["hidden_size"], self.config["output_size"])
        
        # 轨迹校正网络 - 使用2D卷积
        self.correction_net = nn.Sequential(
            # 第一层2D卷积
            nn.Conv2d(1, 32, kernel_size=self.config["kernel_size_conv"], padding=self.config["padding_conv"]),
            nn.AvgPool2d(kernel_size=self.config["kernel_size_pool"], stride=(1, 1), padding=self.config["padding_pool"]),
            
            # 第二层2D卷积
            nn.Conv2d(32, 8, kernel_size=self.config["kernel_size_conv"], padding=self.config["padding_conv"]),
            nn.AvgPool2d(kernel_size=self.config["kernel_size_pool"], stride=(1, 1), padding=self.config["padding_pool"]),
            
            # 最终映射回输出维度
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x, lengths):
        try:
            assert not torch.isnan(x).any(), "NaN in input"
        
            x = x.unsqueeze(1)
            x = self.conv_pre(x)
            x = x.squeeze(1)
            
            x = pad_sequence(x, batch_first=True, padding_value=0.0)

            # 打包
            packed_input = pack_padded_sequence(
                x, 
                lengths,
                batch_first=True, 
                enforce_sorted=True
            )
            
            # GRU处理
            packed_output, _ = self.gru(packed_input)
            
            # 解包
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            output = output.contiguous()
            
            # 注意力
            attention_weights = self.attention(output)
            weighted_output = torch.sum(output * attention_weights, dim=1)
            weighted_output = weighted_output.unsqueeze(1).repeat(1, output.size(1), 1)
            
            # 全连接层
            batch_size, seq_len, hidden_size = output.size()
            output = output + weighted_output  # 残差连接
            output = output.view(-1, hidden_size)
            output = self.fc(output)
            output = output.view(batch_size, seq_len, -1)
            
            correction_input = output.unsqueeze(1) # [batch, 1, seq_len, 2]
            
            correction = self.correction_net(correction_input)
            correction = correction.squeeze(1)  # [batch, seq_len, 2]
            
            output = output + correction  # 残差连接
            
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            raise