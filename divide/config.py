# config.py
import torch

# 硬件配置
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data/frame_standard_delete_g"
SAVED_DATA_PATH = "divide/processed_data.pth"
MODEL_SAVE_PATH = "divide/best_model.pth"

# 模型参数
MODEL_CONFIG = {
    "input_size": 6,
    "hidden_size": 64,
    "num_layers": 2,
    "output_size": 2,
    "dropout": 0.1
}

# 训练参数
TRAIN_CONFIG = {
    "epochs": 200,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "patience": 10,
    "min_delta": 1e-6,
    "batch_size": 4,
    "time_step": 5000,
    "stride": 60
}