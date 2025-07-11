# train.py
from sklearn.model_selection import TimeSeriesSplit
import torch
from .dataset import IMUTrajectoryDataset, collate_fn
from .model import IMUToTrajectoryNet
from .config import *
from torch.utils.data import Subset
# import wandb
import swanlab as wandb
from torch.utils.data import Dataset, DataLoader
from .validate import validate

def train(model, dataloader, optimizer, device):
    model.train()
    sample_losses = {}  
    sample_valid_elements = {}  
    
    total_batches = len(dataloader)
    
    for batch_idx, (inputs, targets, masks, lengths, sample_indices, window_indices) in enumerate(dataloader):
        if batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        inputs = inputs.contiguous().to(device)
        targets = targets.contiguous().to(device)
        masks = masks.contiguous().to(device)
        lengths = lengths.contiguous().to(device)
        
        outputs = model(inputs, lengths)
        
        # 维度匹配
        targets = targets[:, :outputs.shape[1], :].contiguous()
        masks = masks[:, :outputs.shape[1]].contiguous()
        masks = masks.unsqueeze(-1).expand(-1, -1, 2).contiguous()
        
        # 计算每个窗口的损失
        loss_per_element = (outputs - targets) ** 2
        masked_loss = loss_per_element * masks
        
        batch_size = inputs.size(0)
        for i in range(batch_size):
            sample_idx = sample_indices[i].item()  
            
            if sample_idx not in sample_losses:
                sample_losses[sample_idx] = 0
                sample_valid_elements[sample_idx] = 0
            
            # 计算当前样本的损失
            sample_loss = masked_loss[i][masks[i] > 0].sum().item()
            valid_elements = masks[i].sum().item()
            
            sample_losses[sample_idx] += sample_loss
            sample_valid_elements[sample_idx] += valid_elements
        
        # 计算当前批次的平均损失
        current_valid = masks.sum()
        if current_valid > 0:
            loss = masked_loss.sum() / current_valid
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录每个batch的损失
            wandb.log({"batch_loss": loss.item()})
        
        # 显示进度
        # if (batch_idx + 1) % progress_interval == 0:
        #     progress = (batch_idx + 1) / total_batches * 100
        #     avg_loss = sum(sample_losses.values()) / sum(sample_valid_elements.values()) if sample_valid_elements else 0
        #     print(f'Training Progress: {progress:.0f}% - Current Loss: {avg_loss:.8f}')
    
    # 返回所有样本的平均损失
    total_loss = sum(sample_losses.values())
    total_valid = sum(sample_valid_elements.values())
    avg_loss = total_loss / total_valid if total_valid > 0 else 0
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def train_model():
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
    
    # 确保我们有训练集和验证集
    assert train_indices is not None and val_indices is not None
    
    # 计算测试集大小（使用最后10%的数据）
    test_size = int(0.1 * total_size)
    test_start = total_size - test_size
    
    # 调整验证集，不要与测试集重叠
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

    # 学习率调度器
    warmup_steps = TRAIN_CONFIG["warmup_steps"]
    total_steps = TRAIN_CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps - warmup_steps,
        T_mult=1,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience = TRAIN_CONFIG["patience"]
    patience_counter = 0
    min_delta = 1e-6  # 最小改善阈值
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        print("")
        print("Start Training epoch: ", epoch)
        
        # 学习率预热
        if epoch < warmup_steps:
            lr = TRAIN_CONFIG["lr"] * (epoch + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            
        train_loss = train(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE, epoch=epoch, plot=True)

        # 检查是否有显著改善
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_masked_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 学习率调整和早停
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
    test_loss = validate(model, test_loader, DEVICE, plot=True)
    print(f'Final Test Loss: {test_loss:.8f}')
    wandb.log({"final_test_loss": test_loss})

    # 在程序结束时关闭wandb
    wandb.finish()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()