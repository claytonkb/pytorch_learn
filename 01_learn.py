import torch

# * operator performs element-wise multiplication of tensors
# For matrix multiplication, use matmul:
#   https://pytorch.org/docs/stable/torch.html#torch.matmul

a = torch.rand(5,1)
print("a")
print(a)

b = torch.rand(3,1)
print("b")
print(b)

c = torch.rand(3,1)
print("c")
print(c)

print("b*c")
print(b*c)

# Use transpose(0,1) to perform an ordinary matrix transpose. Only works on
# 2d tensors. You can get fancy with this method, but clearer to just select
# a 2d matrix from your n-dim tensor then perform a standard 2d transpose on
# it.

d = a.transpose(0,1)*b # Python tries to be smart and performs a matmul here
print("d=a^T x b")
print(d)


e = b.transpose(0,1)*a # NOTE: e = d^T
print("e=a^T x b")
print(e)


