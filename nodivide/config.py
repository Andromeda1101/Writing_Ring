# config.py
import torch

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            return torch.device('cuda:0')
        except RuntimeError:
            print("CUDA initialization failed, falling back to CPU")
            return torch.device('cpu')
    return torch.device('cpu')

DEVICE = get_device()
DATA_DIR = "data/frame_standard_delete_g" 
SAVED_DATA_PATH = "nodivide/processed_data.pth"     
MODEL_SAVE_PATH = "nodivide/best_model.pth"    
DATA_LENGTH = 10000

# 模型参数
MODEL_CONFIG = {
    "input_size": 6,
    "hidden_size": 512,  
    "num_layers": 3,    
    "output_size": 2,
    "length": DATA_LENGTH,
    "dropout": 0.3
}

# 训练参数
TRAIN_CONFIG = {
    "epochs": 300,
    "lr": 0.0005,      
    "weight_decay": 1e-4,  
    "patience": 10,
    "min_delta": 1e-6,
    "batch_size": 8,   
    "time_step": DATA_LENGTH,
    "stride": 5000,
    "warmup_steps": 20,
    "grad_weight": 2.0,
    "dist_weight": 0.6
}