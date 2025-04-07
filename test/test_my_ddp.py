import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# 配置参数
num_samples = 16            # 总样本数
input_shape = (3, 4, 4)       # 输入维度（通道数，高度，宽度）
batch_size = 4              # 批大小
num_classes = 10            # 分类类别数

# 1. 定义随机数据集
class RandomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(num_samples, *input_shape)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. 定义浅层神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# 3. 初始化分布式环境
def setup(rank, world_size):
    """
    初始化进程组，backend使用gloo（因为没有安装NCCL）
    """
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    """
    销毁进程组
    """
    dist.destroy_process_group()

# 4. 训练函数，每个进程运行一份训练代码
def train(rank, world_size):
    print(f"在进程 {rank} 上启动DDP训练")
    setup(rank, world_size)

    # 为当前进程设置对应的GPU设备（假设每个进程对应一个GPU）
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # 构建模型并将模型移动到对应的GPU
    model = SimpleModel().to(device)
    # 包装为DDP模型
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # 构建数据集和分布式采样器，保证每个进程看到不同的数据子集
    dataset = RandomDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    ddp_model.train()
    print(f"进程 {rank} 上训练开始...")
    for inputs, labels in dataloader:
        # 将数据移动到对应的GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'进程 {rank} - 当前批次损失: {loss.detach().item():.4f}')

    cleanup()

if __name__ == '__main__':
    # 总进程数，即GPU数量
    world_size = 4
    # 使用多进程启动训练，每个进程在一个GPU上进行训练
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
