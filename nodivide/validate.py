import os
import torch
from .model import IMUToTrajectoryNet
from .config import *
import swanlab as wandb
import tqdm
from .utils import speed_loss, traject_loss, draw_trajectory_plots

def validate(model, dataloader, epoch=0, plot=True, is_test=False):
    if model is None:
        model = IMUToTrajectoryNet()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    samples_pred = {}
    samples_targ = {}
    device = DEVICE
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, sample_indices, window_indices) in tqdm.tqdm(enumerate(dataloader)):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(inputs)

            valid_num = masks.sum()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2)
            loss = speed_loss(outputs, targets, masks)
            traj_loss = traject_loss(outputs * masks, targets)

            total_loss += loss + traj_loss
            
            if plot and epoch is not None and epoch % 10 == 0:
                sample_idx_list = sample_indices.cpu().numpy()
                window_idx_list = window_indices.cpu().numpy()

                for idx in sample_idx_list:
                    if idx not in samples_pred:
                        samples_pred[idx] = {}
                        samples_targ[idx] = {}

                masked_outputs = (outputs * masks).cpu().numpy()
                masked_targets = (targets * masks).cpu().numpy()
                
                for i, (s_idx, w_idx) in enumerate(zip(sample_idx_list, window_idx_list)):
                    samples_pred[s_idx][w_idx] = masked_outputs[i]
                    samples_targ[s_idx][w_idx] = masked_targets[i]
    
    # 所有样本的平均损失
    avg_loss = total_loss / len(dataloader)
    wandb.log({"val_loss": avg_loss})
    
    if plot and epoch is not None and epoch % 10 == 0:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
        for sample_idx, pred_windows in tqdm.tqdm(samples_pred.items()):
            if not is_test and (sample_idx // 10 > 91 or sample_idx // 10 < 90): continue
            sample_dir = os.path.join(plot_dir, f'sample_{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            window_indices = sorted(pred_windows.keys())
            targ_windows = samples_targ[sample_idx]
            for window_idx in window_indices:
                try:
                    img_path = os.path.join('trajectory_plots',f'sample_{sample_idx}', f'window_{window_idx}_epoch_{epoch}.png')
                    draw_trajectory_plots(pred_windows, targ_windows, epoch, window_idx, sample_idx, img_path)
                    if os.path.exists(img_path):
                        wandb.log({f"trajectory_plot_{sample_idx}_{window_idx}": wandb.Image(img_path)})
                except Exception as e:
                    print(f"Error logging trajectory plot: {e}")
    
    return avg_loss

