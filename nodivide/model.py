# model.py
import torch.nn as nn
from .config import MODEL_CONFIG
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IMUToTrajectoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MODEL_CONFIG
        
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

    def forward(self, x, lengths):
        try:
            assert not torch.isnan(x).any(), "NaN in input"
            assert (lengths > 0).all(), "Non-positive lengths"
            
            x = x.contiguous()
            lengths = lengths.cpu().contiguous()
            
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
            
            return output.contiguous()
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            print(f"Lengths: {lengths}")
            raise