from matplotlib import pyplot as plt
from .model import *
from torch.utils.data import Dataset, DataLoader, TensorDataset


def augment_data(generator, original_vel, num_samples):
    """使用生成器增强数据"""
    generator.eval()
    augmented_imu = []
    
    with torch.no_grad():
        # 分批生成
        for i in range(0, num_samples, GANConfig.batch_size):
            batch_size = min(GANConfig.batch_size, num_samples - i)
            
            # 创建噪声和条件
            z = torch.randn(batch_size, GANConfig.noise_dim, device=DEVICE)
            conditions = original_vel[i:i+batch_size].to(DEVICE)
            
            # 生成IMU数据
            fake_imu = generator(z, conditions)
            augmented_imu.append(fake_imu.cpu())
    
    return torch.cat(augmented_imu, dim=0)

def enhance_dataset_gan(imu_samples, vel_samples):

    generator = Generator()
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.to(DEVICE)

    augmented_imu = augment_data(generator, vel_samples, len(vel_samples))

    # 组合原始和增强数据
    combined_imu = torch.cat([imu_samples, augmented_imu], dim=0)
    combined_vel = torch.cat([vel_samples, vel_samples.clone()], dim=0)

    return combined_imu, combined_vel

def vae_loss_function(x, recon_x, mu, logvar):
    # 这里要计算ELBO(也就是论文中的$\mathcal{L}$)，但是由于论文中的目标是最大化ELBO，pytorch中是最小化loss，所以这里实际计算的是-ELBO
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # 计算KL散度
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') # 计算重构误差，对应论文中的$-\log p_{\theta}(x|z)$，注意BCE loss本身前面有个负号
    return KLD + BCE

def draw_vae_samples(model, dataloader):
    recon_batch = []
    with torch.no_grad():
        for v in dataloader:
            z = model.encoder(v)
            recon_batch = model.decode(z)
            recon_batch = recon_batch.view(-1, -1, 2).cpu().numpy()

    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, len(recon_batch), i+1)
        plt.imshow(recon_batch[i], cmap='gray')
        plt.axis('off')
    plt.show()