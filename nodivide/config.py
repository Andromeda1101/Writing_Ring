# config.py
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data/frame_standard_delete_g" 
SAVED_DATA_PATH = "nodivide/processed_data.pth"     
MODEL_SAVE_PATH = "nodivide/best_model.pth"    

# 模型参数
MODEL_CONFIG = {
    "input_size": 6,
    "hidden_size": 128,  
    "num_layers": 3,    
    "output_size": 2,
    "dropout": 0.2     
}

# 训练参数
TRAIN_CONFIG = {
    "epochs": 200,
    "lr": 0.0005,      
    "weight_decay": 1e-4,  
    "patience": 15,
    "min_delta": 1e-6,
    "batch_size": 32,   
    "time_step": 15000,
    "stride": 10000,
    "warmup_steps": 10
}