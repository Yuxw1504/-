#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:52:47 2024

@author: yxw
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 测试是否成功导入torch
print("PyTorch version:", torch.__version__)

# 创建一个简单的张量并进行基本操作
x = torch.rand(5, 3)
print("Random tensor x:\n", x)

y = torch.ones(5, 3)
print("Tensor y (ones):\n", y)

z = x + y
print("Result of x + y:\n", z)

# 构建一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNN()
print("Model structure:\n", model)

# 创建一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建一个简单的输入张量
input_tensor = torch.rand(5, 3)
print("Input tensor:\n", input_tensor)

# 前向传播
output = model(input_tensor)
print("Model output:\n", output)

# 定义损失函数
criterion = nn.MSELoss()

# 创建目标张量
target = torch.rand(5, 1)

# 计算损失
loss = criterion(output, target)
print("Loss:\n", loss)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Test completed successfully!")


