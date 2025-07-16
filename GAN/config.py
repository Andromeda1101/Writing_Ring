import torch
from nodivide.config import TRAIN_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAN_DATA_PATH = "gan_processed_data.pth"
SAMPLES_PATH = "generate_samples"
GENERATOR_PATH = "imu_generator.pth"
DISCRIMINATOR_PATH = "imu_discriminator.pth"
SEQ_LENGTH = TRAIN_CONFIG.time_step
STRIDE = TRAIN_CONFIG.stride

class GANConfig:
    seq_length = SEQ_LENGTH      
    stride = STRIDE
    imu_dim = 6           
    vel_dim = 2           
    noise_dim = 128      
    hidden_dim = 64      
    batch_size = 64     
    lr = 0.0002          
    epochs = 500
    sample_interval = 500

class VAEConfig:
    vae_dir = "vae"
    plots_dir = "vae_plots"
    model_path = "vae_best_model.pth"
    final_model_path = "vae_final_model.pth"
    full_length = SEQ_LENGTH
    full_stride = STRIDE
    seq_len = 100
    stride = 50
    input_dim = 2
    hidden_dim = 64 
    latent_dim = 32 
    dropout = 0.2
    num_layers = 2
    epochs = 400
    lr = 0.0005  
    test_freq = 10
    batch_size = 64  
    patience = 15  
    weight_decay = 1e-3 
    kld_weight = 0.7
