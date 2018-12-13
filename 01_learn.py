import torch
import numpy as np

# NOTE: * operator performs element-wise multiplication of tensors
# For matrix multiplication, use matmul:
#   https://pytorch.org/docs/stable/torch.html#torch.matmul

a = torch.rand(3,1)
b = torch.rand(3,1).transpose(0,1)

print("a")
print(a)

print("b")
print(b)

c = torch.matmul(a, b)
print("c=a*b")
print(c.numpy())

d = torch.matmul(b, a)
print("d=b*a")
print(d.numpy())

# Use transpose(0,1) to perform an ordinary matrix transpose. Only works on
# 2d tensors. You can get fancy with this method, but clearer to just select
# a 2d matrix from your n-dim tensor then perform a standard 2d transpose on
# it.

print("c^T")
print(c.transpose(0,1).numpy())


