from sklearn.model_selection import TimeSeriesSplit
from nodivide.dataset import IMUTrajectoryDataset
import torch

if __name__ == "__main__":
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
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, range(test_start, total_size))
    
