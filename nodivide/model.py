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
        self.fc = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, x, lengths):
        try:
            assert not torch.isnan(x).any(), "NaN in input"
            assert (lengths > 0).all(), "Non-positive lengths"
            
            x = x.contiguous()
            lengths = lengths.cpu().contiguous()
            
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
            
            # 全连接层
            batch_size, seq_len, hidden_size = output.size()
            output = output.view(-1, hidden_size)
            output = self.fc(output)
            output = output.view(batch_size, seq_len, -1)
            
            return output.contiguous()
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}")
            print(f"Lengths: {lengths}")
            raise