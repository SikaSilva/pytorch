import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 配置参数
num_samples = 16       # 总样本数
input_shape = (3, 4, 4)  # 输入维度（通道数，高度，宽度）
batch_size = 4         # 批大小
num_classes = 10       # 分类类别数

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
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# 3. 初始化模型并包装为DataParallel
model = SimpleModel()
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()  # 使用4个GPU

# 4. 准备数据加载器
dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 6. 训练循环（单个epoch）
model.train()
print("training started...")
for inputs, labels in dataloader:
    # 将数据移动到GPU（默认会分配到第一个GPU，DataParallel会自动分发）
    inputs = inputs.cuda()
    labels = labels.cuda()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'curr batch loss: {loss.detach().item():.4f}')

print("training done...")
