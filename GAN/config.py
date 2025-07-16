import torch
from nodivide.config import TRAIN_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = TRAIN_CONFIG.time_step
STRIDE = TRAIN_CONFIG.stride

class GANConfig:
    gan_dir = "gan"
    plots_dir = "gan_plots"
    generator_path = "imu_generator.pth"
    discriminator_path = "imu_discriminator.pth"
    full_length = SEQ_LENGTH
    full_stride = STRIDE
    seq_len = 100
    stride = 50
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
    latent_dim = 128 
    dropout = 0.2
    num_layers = 2
    epochs = 400
    lr = 0.0005  
    test_freq = 10
    batch_size = 64  
    patience = 15  
    weight_decay = 1e-3 
    kld_weight = 0.7
