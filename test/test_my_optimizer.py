import torch
import torch.nn as nn
import torch.nn.functional as F


# class ToyModel(nn.Module):

#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def print_module_parameters(sentence: str, m: nn.Module):
#     print('\n')
#     print(sentence)
#     for n, p in m.named_parameters():
#         print('name = {}\n{}\n'.format(n, p))
#     print('\n')

# net = ToyModel()
# optimizer = torch.optim.SGD(params=net.parameters(), lr=1)
# optimizer.zero_grad()
# input = torch.randn(10, 10)
# outputs = net(input)
# outputs.backward(outputs)

# print_module_parameters('before autograd:', net)
# optimizer.step()

# print_module_parameters('after autograd', net)


# print('Model.state_dict:')
# for param_tensor in net.state_dict():
#     print(param_tensor, '\t', net.state_dict()[param_tensor].size())

# # print optimizer's state_dict
# print('Optimizer.state_dict:')
# for var_name in optimizer.state_dict():
#     print(var_name, '\t', optimizer.state_dict()[var_name])


from math import pi
import torch.optim

x = torch.tensor([pi / 2, pi / 3], requires_grad=True)
optimizer = torch.optim.SGD([x,], lr=0.2, momentum=0.5)

for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

        for var_name in optimizer.state_dict():
            print(var_name, '\t', optimizer.state_dict()[var_name])
    f = -((x.sin()**3).sum())**3
