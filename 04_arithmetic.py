import torch
import numpy as np

# NOTE: torch.mm() is a less fancy version of torch.matmul() ... may be better
# for situations where you want to know exactly what pytorch is doing behind
# the scenes
#
# https://pytorch.org/docs/stable/torch.html#torch.addmm
# This could be handy for manual implementation of NN-layer

#torch.tensor([[1., -1.], [1., -1.]])
#tensor([[ 1.0000, -1.0000],
#        [ 1.0000, -1.0000]])
#>>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

x = torch.rand(3,1)
print("x")
print(x)

y = torch.rand(3,1).transpose(0,1)
print("y")
print(y)

z = torch.matmul(x, y)
print("z=x*y")
print(z.numpy())

w = torch.matmul(y, x)
print("w=y*x")
print(w.numpy())

# Use transpose(0,1) to perform an ordinary matrix transpose. Only works on
# 2d tensors. You can get fancy with this method, but clearer to just select
# a 2d matrix from your n-dim tensor then perform a standard 2d transpose on
# it.

print("z^T")
print(z.transpose(0,1).numpy())

w = torch.matmul(z, x)
print("w=z*x")
print(w.numpy())

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
print("a")
print(a)

# creating this tensor_f64 thingy seems questionable... ?
tensor_f64 = torch.tensor((), dtype=torch.float64)
b = tensor_f64.new_full((2,3), 2.71828)
print("b")
print(b)


# element-wise sum
c = b.add(a)
print("c")
print(c)

# element-wise product (Hadamard product)
d = b.mul(a)
print("d")
print(d)

e = torch.matmul(d,d.transpose(0,1)) # size = nrows^2
#e = torch.matmul(d.transpose(0,1),d) # size = ncols^2
print("e=d*d^T")
print(e)

print("eigenvalues of e")
print(e.eig())

f = torch.tensor([1,2,3])
g = torch.tensor([4,5,6])
print("f")
print(f)
print("g")
print(g)

h = torch.dot(f, g)
print("h=dot(f,g)")
print(h)




