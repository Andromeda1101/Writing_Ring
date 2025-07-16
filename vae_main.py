import argparse
from GAN.train import train_vae_model
from GAN.config import VAEConfig

def parse_args():
    parser = argparse.ArgumentParser(description='VAE Training Parameters')
    parser.add_argument('--vae_dir', type=str, default=VAEConfig.vae_dir,
                       help='Directory for VAE model')
    parser.add_argument('--plots_dir', type=str, default=VAEConfig.plots_dir,
                       help='Directory for saving plots')
    parser.add_argument('--model_path', type=str, default=VAEConfig.model_path,
                       help='Path to save best model')
    parser.add_argument('--final_model_path', type=str, default=VAEConfig.final_model_path,
                       help='Path to save final model')
    parser.add_argument('--full_length', type=int, default=VAEConfig.full_length,
                       help='Full sequence length')
    parser.add_argument('--full_stride', type=int, default=VAEConfig.full_stride,
                       help='Full sequence stride')
    parser.add_argument('--seq_len', type=int, default=VAEConfig.seq_len,
                       help='Training sequence length')
    parser.add_argument('--stride', type=int, default=VAEConfig.stride,
                       help='Training sequence stride')
    parser.add_argument('--input_dim', type=int, default=VAEConfig.input_dim,
                       help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=VAEConfig.hidden_dim,
                       help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=VAEConfig.latent_dim,
                       help='Latent dimension')
    parser.add_argument('--dropout', type=float, default=VAEConfig.dropout,
                       help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=VAEConfig.num_layers,
                       help='Number of GRU layers')
    parser.add_argument('--epochs', type=int, default=VAEConfig.epochs,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=VAEConfig.lr,
                       help='Learning rate')
    parser.add_argument('--test_freq', type=int, default=VAEConfig.test_freq,
                       help='Testing frequency')
    parser.add_argument('--batch_size', type=int, default=VAEConfig.batch_size,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=VAEConfig.patience,
                       help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=VAEConfig.weight_decay,
                       help='Weight decay for optimizer')
    parser.add_argument('--kld_weight', type=float, default=VAEConfig.kld_weight,
                       help='Weight for KLD loss')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    train_vae_model(config=args)