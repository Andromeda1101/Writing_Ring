import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PATH = "generate_samples"
GENERATOR_PATH = "GAN/imu_generator.pth"
DISCRIMINATOR_PATH = "GAN/imu_discriminator.pth"

class GANConfig:
    seq_length = 10000      
    imu_dim = 6           
    vel_dim = 2           
    noise_dim = 128      
    hidden_dim = 64      
    batch_size = 64     
    lr = 0.0002          
    epochs = 500
    sample_interval = 500