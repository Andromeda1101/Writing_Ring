import torch
from nodivide.config import TRAIN_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAN_DATA_PATH = "gan_processed_data.pth"
VAE_DATA_PATH = "vae_processed_data.pth"
SAMPLES_PATH = "generate_samples"
GENERATOR_PATH = "imu_generator.pth"
DISCRIMINATOR_PATH = "imu_discriminator.pth"
VAE_PICT_DIR = "vae_pict"
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
    full_length = SEQ_LENGTH
    full_stride = STRIDE
    seq_len = 100
    stride = 50
    input_dim = 2
    hidden_dim = 128
    latent_dim = 256
    epochs = 100
    lr = 0.001
    test_freq = 10
    batch_size = 16