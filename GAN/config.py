import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/frame_standard_delete_g" 
GAN_DATA_PATH = "gan_processed_data.pth"
VAE_DATA_PATH = "vae_processed_data.pth"
SAMPLES_PATH = "generate_samples"
GENERATOR_PATH = "imu_generator.pth"
DISCRIMINATOR_PATH = "imu_discriminator.pth"
VAE_PICT_DIR = "vae_pict"
SEQ_LENGTH = 10000
STRIDE = 5000

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
    seq_length = SEQ_LENGTH
    stride = STRIDE
    input_dim = 2 * SEQ_LENGTH
    hidden_dim = 4096
    latent_dim = 1024
    epochs = 300
    lr = 0.001
    test_freq = 20
    batch_size = 16