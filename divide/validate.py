import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb

from .config import MODEL_SAVE_PATH, DEVICE

from .train import IMUToTrajectoryNet


def validate(model, dataloader, device, epoch=None, plot=False):
    if model is None:
        model = IMUToTrajectoryNet()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(DEVICE)
    model.eval()
    sample_losses = {}
    sample_valid_elements = {}
    
    # Create plot directory
    if plot:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths, sample_indices) in enumerate(dataloader):
            inputs = inputs.contiguous().to(device)
            targets = targets.contiguous().to(device)
            lengths = lengths.contiguous().to(device)
            
            outputs = model(inputs, lengths)
            targets = targets[:, :outputs.shape[1], :].contiguous()
            
            loss_per_element = (outputs - targets) ** 2
            
            sample_idx = sample_indices[0].item()
            
            if sample_idx not in sample_losses:
                sample_losses[sample_idx] = 0
                sample_valid_elements[sample_idx] = 0
            
            current_loss = loss_per_element.sum().item()
            current_length = lengths[0].item()
            
            sample_losses[sample_idx] += current_loss
            sample_valid_elements[sample_idx] += current_length * 2
            
            # Plot trajectory comparison
            if plot and batch_idx < 5:
                pred_trajectory = outputs[0].cpu().numpy()
                true_trajectory = targets[0].cpu().numpy()
                valid_length = lengths[0].cpu().numpy()
                
                # Get valid trajectory points
                pred_valid = pred_trajectory[:valid_length]
                true_valid = true_trajectory[:valid_length]
                time_valid = np.arange(len(pred_valid))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot X coordinate
                ax1.plot(time_valid, pred_valid[:, 0], 'r-', label='Predicted', linewidth=2)
                ax1.plot(time_valid, true_valid[:, 0], 'b-', label='Ground Truth', linewidth=2)
                ax1.set_title('X Coordinate over Time')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('X Position')
                ax1.legend()
                ax1.grid(True)
                
                # Plot Y coordinate
                ax2.plot(time_valid, pred_valid[:, 1], 'r-', label='Predicted', linewidth=2)
                ax2.plot(time_valid, true_valid[:, 1], 'b-', label='Ground Truth', linewidth=2)
                ax2.set_title('Y Coordinate over Time')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Y Position')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                
                epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
                plt.savefig(os.path.join(plot_dir, f'trajectory_{batch_idx}{epoch_str}.png'))
                plt.close()
    
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else float('inf')
    wandb.log({"val_loss": avg_loss})
    
    # Upload trajectory plots to wandb
    if plot and epoch is not None:
        for batch_idx in range(min(5, len(dataloader))):
            try:
                img_path = os.path.join('trajectory_plots', f'trajectory_{batch_idx}_epoch_{epoch}.png')
                if os.path.exists(img_path):
                    wandb.log({f"trajectory_plot_{batch_idx}": wandb.Image(img_path)})
            except Exception as e:
                print(f"Error logging trajectory plot: {e}")
    
    return avg_loss