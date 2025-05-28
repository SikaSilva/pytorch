import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import functools

# 定义原始模型


class DoubleLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(2000, 2000)
        self.linear1 = nn.Linear(2000, 2000)

    def forward(self, x):
        y0 = self.linear0(x)
        y1 = self.linear1(y0)
        return y1


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl0 = DoubleLinearModule()
        self.dl1 = DoubleLinearModule()
        self.dl2 = DoubleLinearModule()

    def forward(self, x):
        y0 = self.dl0(x)
        y1 = self.dl1(y0)
        y2 = self.dl2(y1)
        return y2

# 自定义数据集


class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.input_size)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12300'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def custom_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    """
    自定义FSDP包装策略，决定是否在特定模块处进行包装

    参数:
    - module: 当前考虑的模块
    - recurse: 是否递归到子模块
    - unwrapped_params: 当前尚未包装的参数数量

    返回:
    - bool: 是否在此模块处进行包装
    """
    # 如果是递归阶段，继续向下探索
    if recurse:
        return True

    # 仅在DoubleLinearModule类型且参数超过100万的模块处包装
    return isinstance(module, DoubleLinearModule)


def train(rank, world_size, num_epochs=5):
    setup(rank, world_size)

    # 初始化模型
    model = SimpleModel()
    model = FSDP(model,
                 auto_wrap_policy=custom_auto_wrap_policy,
                 device_id=rank,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 准备数据
    dataset = RandomDataset(num_samples=20, input_size=2000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs = batch.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = outputs.sum()
            loss.backward()
            optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.detach().item()}")

    cleanup()


def main():
    world_size = 2
    torch.multiprocessing.spawn(
        train,
        args=(world_size, 5),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
