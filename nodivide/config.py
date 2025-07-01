# config.py
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data/frame_standard_delete_g" 
SAVED_DATA_PATH = "nodivide/processed_data.pth"     
MODEL_SAVE_PATH = "nodivide/best_model.pth"    
DATA_LENGTH = 15000

# 模型参数
MODEL_CONFIG = {
    "input_size": 6,
    "hidden_size": 128,  
    "num_layers": 3,    
    "output_size": 2,
    "kernel_size_conv1": (256 + 1, 2 + 1),
    "kernel_size_pool1": (256 + 1, 1),
    "padding_conv1": (128, 1),
    "padding_pool1": (128, 0),
    "kernel_size_conv2": (128 + 1, 2 + 1),
    "kernel_size_pool2": (128 + 1, 1),
    "padding_conv2": (64, 1),
    "padding_pool2": (64, 0),
    "length": DATA_LENGTH,
    "dropout": 0.2     
}

# 训练参数
TRAIN_CONFIG = {
    "epochs": 300,
    "lr": 0.0005,      
    "weight_decay": 1e-4,  
    "patience": 10,
    "min_delta": 1e-6,
    "batch_size": 32,   
    "time_step": DATA_LENGTH,
    "stride": 10000,
    "warmup_steps": 10,
    "grad_weight": 0.6,
    "dist_weight": 0.6
}