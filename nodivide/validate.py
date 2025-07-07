import os
import torch
from .model import IMUToTrajectoryNet
from .config import *
import swanlab as wandb
import tqdm
from .utils import velocity_loss, traject_loss, draw_trajectory_plots

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
    pict_num = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, masks, sample_indices, window_indices) in tqdm.tqdm(enumerate(dataloader)):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(inputs)

            valid_num = masks.sum()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2)
            loss = velocity_loss(outputs, targets, masks)
            traj_loss = traject_loss(outputs * masks, targets)

            total_loss += loss + traj_loss
            
            if plot and epoch is not None and epoch % 10 == 0:
                if pict_num < 5: 
                    for pred, targ, sample_idx, window_idx in zip(outputs, targets, sample_indices, window_indices):
                        if pict_num >= 5: continue
                        pict_num += 1
                        if sample_idx not in samples_pred:
                            samples_pred[sample_idx] = {}
                            samples_targ[sample_idx] = {}
                        samples_pred[sample_idx][window_idx] = pred.cpu().numpy()
                        samples_targ[sample_idx][window_idx] = targ.cpu().numpy()
                    
    
    # 所有样本的平均损失
    avg_loss = total_loss / len(dataloader)
    wandb.log({"val_loss": avg_loss})
    
    if plot and epoch is not None and epoch % 10 == 0:
        plot_dir = 'trajectory_plots'
        os.makedirs(plot_dir, exist_ok=True)
        pict_num = 0
        for sample_idx, pred_windows in tqdm.tqdm(samples_pred.items()):
            sample_dir = os.path.join(plot_dir, f'sample_{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            window_indices = sorted(pred_windows.keys())
            targ_windows = samples_targ[sample_idx]
            for window_idx in window_indices:
                try:
                    img_path = os.path.join('trajectory_plots',f'sample_{sample_idx}', f'window_{window_idx}_epoch_{epoch}.png')
                    draw_trajectory_plots(pred_windows[window_idx], targ_windows[window_idx], epoch, window_idx, sample_idx, img_path)
                    if os.path.exists(img_path):
                        wandb.log({f"trajectory_plot_{sample_idx}_{window_idx}": wandb.Image(img_path)})
                except Exception as e:
                    print(f"Error logging trajectory plot: {e}")
    
    return avg_loss

