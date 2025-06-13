import torch

# 创建6个3x3张量，每个元素值为编号，并启用梯度跟踪
t1 = torch.full((3, 3), 1.0, dtype=torch.float32, requires_grad=True)
t2 = torch.full((3, 3), 2.0, dtype=torch.float32, requires_grad=True)
t3 = torch.full((3, 3), 3.0, dtype=torch.float32, requires_grad=True)
t4 = torch.full((3, 3), 4.0, dtype=torch.float32, requires_grad=True)
t5 = torch.full((3, 3), 5.0, dtype=torch.float32, requires_grad=True)
t6 = torch.full((3, 3), 6.0, dtype=torch.float32, requires_grad=True)

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

# import torch
# from torchviz import make_dot
# import os

# # 创建目录（如果不存在）
# # os.makedirs("/root/workspace/pytorch", exist_ok=True)

# # 创建张量
# t1 = torch.full((3, 3), 1.0, requires_grad=True)
# t2 = torch.full((3, 3), 2.0, requires_grad=True)
# t3 = torch.full((3, 3), 3.0, requires_grad=True)
# t4 = torch.full((3, 3), 4.0, requires_grad=True)
# t5 = torch.full((3, 3), 5.0, requires_grad=True)
# t6 = torch.full((3, 3), 6.0, requires_grad=True)

# # 前向计算
# s1 = t1 + t2
# s2 = t3 + t4
# s3 = t5 + t6
# m1 = s1 @ s2
# y = m1 @ s3
# y_sum = y.sum()

# # 生成计算图（包含反向传播）
# dot = make_dot(
#     y_sum, 
#     params={
#         't1': t1, 't2': t2, 't3': t3,
#         't4': t4, 't5': t5, 't6': t6,
#         's1': s1, 's2': s2, 's3': s3,
#         'm1': m1, 'y': y
#     },
#     show_attrs=True,
#     show_saved=True
# )

# # 保存为图片文件
# dot.format = 'jpg'
# dot.render(filename='/root/workspace/pytorch/temp', cleanup=True)

# # 执行反向传播
# y_sum.backward()

# # 打印梯度（可选）
# print("t1.grad:\n", t1.grad)
# print("\nt2.grad:\n", t2.grad)
# print("\nt3.grad:\n", t3.grad)
# print("\nt4.grad:\n", t4.grad)
# print("\nt5.grad:\n", t5.grad)
# print("\nt6.grad:\n", t6.grad)
