# train.py
import sys
import os
from sklearn.model_selection import TimeSeriesSplit
import torch
import wandb
from torch.utils.data import DataLoader

from divide import validate
from .config import *
from .dataset import IMUTrajectoryDataset, collate_fn
from .model import IMUToTrajectoryNet

def train(model, dataloader, optimizer, device):
    model.train()
    sample_losses = {}  
    sample_valid_elements = {}  
    
    total_batches = len(dataloader)
    
    for batch_idx, (inputs, targets, lengths, sample_indices) in enumerate(dataloader):
        if batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
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
        
        loss = loss_per_element.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record batch loss
        wandb.log({"batch_loss": loss.item()})
    
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else 0
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def train_model():
    # 初始化wandb
    wandb.init(project="imu-trajectory", config={**MODEL_CONFIG, **TRAIN_CONFIG})

    # 数据集
    print(f'\nLoading data')
    full_dataset = IMUTrajectoryDataset()
    total_size = len(full_dataset)
    
    tscv = TimeSeriesSplit(n_splits=5)  
    
    # 获取最后一次划分的索引
    train_indices = None
    val_indices = None
    for train_idx, val_idx in tscv.split(range(total_size)):
        train_indices = train_idx
        val_indices = val_idx
    
    assert train_indices is not None and val_indices is not None
    
    test_size = int(0.1 * total_size)
    test_start = total_size - test_size
    
    # 不与测试集重叠
    val_indices = val_indices[val_indices < test_start]
    
    # 创建数据集
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
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],  
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"],  
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    model = IMUToTrajectoryNet().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    # 早停
    best_val_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0
    min_delta = 1e-6  # 最小改善阈值
    

    for epoch in range(TRAIN_CONFIG["epochs"]):
        print("")
        print("Start Training epoch: ", epoch)
        train_loss = train(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE, epoch=epoch, plot=True)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_masked_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
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
    
    test_loss = validate(model, test_loader, DEVICE, plot=True)
    print(f'Final Test Loss: {test_loss:.8f}')
    wandb.log({"final_test_loss": test_loss})

    # Close wandb
    wandb.finish()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()