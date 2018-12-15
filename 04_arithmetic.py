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

