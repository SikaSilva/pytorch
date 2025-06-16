import torch

# 创建6个3x3张量，每个元素值为编号，并启用梯度跟踪
t1 = torch.full((3, 3), 1.0, dtype=torch.float32, requires_grad=True, device="cuda")
t2 = torch.full((3, 3), 2.0, dtype=torch.float32, requires_grad=True, device="cuda")
t3 = torch.full((3, 3), 3.0, dtype=torch.float32, requires_grad=True, device="cuda")
t4 = torch.full((3, 3), 4.0, dtype=torch.float32, requires_grad=True, device="cuda")
t5 = torch.full((3, 3), 5.0, dtype=torch.float32, requires_grad=True, device="cuda")
t6 = torch.full((3, 3), 6.0, dtype=torch.float32, requires_grad=True, device="cuda")

# 计算表达式 y = (1+2) @ (3+4) @ (5+6)
y = (t1 + t2) @ (t3 + t4) @ (t5 + t6)

# 对结果求和并反向传播
y_sum = y.sum()
y_sum.backward()

# 打印各张量的梯度
print("t1.grad:\n", t1.grad)
print("\nt2.grad:\n", t2.grad)
print("\nt3.grad:\n", t3.grad)
print("\nt4.grad:\n", t4.grad)
print("\nt5.grad:\n", t5.grad)
print("\nt6.grad:\n", t6.grad)