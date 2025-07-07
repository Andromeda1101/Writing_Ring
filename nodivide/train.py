# train.py
from sklearn.model_selection import TimeSeriesSplit
import torch
from .dataset import IMUTrajectoryDataset, collate_fn
from .model import IMUToTrajectoryNet
from .config import *
from torch.utils.data import Subset
# import wandb
import swanlab as wandb
import tqdm
from torch.utils.data import Dataset, DataLoader
from .validate import validate
import numpy as np
from .utils import velocity_loss, traject_loss

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets, masks, sample_indices, window_indices) in tqdm.tqdm(enumerate(dataloader)):
        try:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(inputs)
            # if torch.isnan(outputs).any():
            #     raise Exception("NaN detected in outputs")
            valid_num = masks.sum()
            masks = masks.unsqueeze(-1).expand(-1, -1, 2)
            outputs = outputs * masks
            loss = velocity_loss(outputs, targets, masks)
            # Check for NaN values in loss
            # if torch.isnan(loss).any():
            #     raise Exception("NaN detected in loss")
            traj_loss = traject_loss(outputs * masks, targets)
            
            optimizer.zero_grad()
            loss = loss + traj_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.sum()
            wandb.log({"batch_loss": loss.sum() / valid_num})
            
            # 显示进度
            # if (batch_idx + 1) % progress_interval == 0:
            #     progress = (batch_idx + 1) / total_batches * 100
            #     avg_loss = sum(sample_losses.values()) / sum(sample_valid_elements.values()) if sample_valid_elements else 0
            #     print(f'Training Progress: {progress:.0f}% - Current Loss: {avg_loss:.8f}')
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            raise

    # 返回所有样本的平均损失
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss})
    return avg_loss




def train_model():
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        
    # 初始化wandb
    wandb.init(project="imu-trajectory", config={**MODEL_CONFIG, **TRAIN_CONFIG})

    # 加载数据集
    print(f'\nLoading data')
    full_dataset = IMUTrajectoryDataset()
    total_size = len(full_dataset)
    
    # 使用TimeSeriesSplit进行数据集划分
    tscv = TimeSeriesSplit(n_splits=5)  # 5折交叉验证
    
    # 获取最后一次划分的索引
    train_indices = None
    val_indices = None
    for train_idx, val_idx in tscv.split(range(total_size)):
        train_indices = train_idx
        val_indices = val_idx
    
    assert train_indices is not None and val_indices is not None
    
    test_size = int(0.1 * total_size)
    test_start = total_size - test_size
    
    val_indices = val_indices[val_indices < test_start]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, range(test_start, total_size))
    
    print(f'\nSplitting dataset:')
    print(f'Total samples: {total_size}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Testing samples: {len(test_dataset)}')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = IMUToTrajectoryNet()
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # 学习率调度器
    warmup_steps = TRAIN_CONFIG["warmup_steps"]
    total_steps = TRAIN_CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_CONFIG["lr"],
        epochs=TRAIN_CONFIG["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.05, 
        anneal_strategy='cos',
        div_factor=25,  
        final_div_factor=1e4
    )

    best_val_loss = float('inf')
    patience = TRAIN_CONFIG["patience"]
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        print("")        
        
        train_loss = train(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss = validate(model, val_loader, epoch=epoch, plot=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_masked_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        print(f'Epoch [{epoch+1}/{TRAIN_CONFIG["epochs"]}], '
              f'Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "patience_counter": patience_counter
        })
    
    # model.load_state_dict(torch.load('best_masked_model.pth'))
    test_loss = validate(model, test_loader, plot=True, is_test=True)
    print(f'Final Test Loss: {test_loss:.8f}')
    wandb.log({"final_test_loss": test_loss})

    # 在程序结束时关闭wandb
    wandb.finish()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()