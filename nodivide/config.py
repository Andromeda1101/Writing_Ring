# config.py
import torch
import os

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            return torch.device('cuda:0')
        except RuntimeError:
            print("CUDA initialization failed, falling back to CPU")
            return torch.device('cpu')
    return torch.device('cpu')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = get_device()
DATA_DIR = "data/frame_standard_delete_g" 
SAVED_DATA_PATH = os.path.join(CURRENT_DIR,"../processed_data.pth" )    
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR,"../best_model.pth")
FINAL_SAVE_PATH = os.path.join(CURRENT_DIR,"../final_model.pth") 
DATA_LENGTH = 10000

# 模型参数
class MODEL_CONFIG:
    input_size = 6
    hidden_size = 128
    num_layers = 3
    output_size = 2
    length = DATA_LENGTH
    dropout = 0.3

# 训练参数
class TRAIN_CONFIG:
    epochs = 300
    lr = 0.005    
    weight_decay = 1e-4 
    patience = 10
    min_delta = 1e-6
    batch_size = 8  
    time_step = DATA_LENGTH
    stride = 5000
    warmup_steps = 10
    rel_weight=0.6
    length_weight=0.4
    abs_weight=0.2
    dir_weight=0.4
    data_size = 1.0
