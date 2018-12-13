import torch
import numpy as np

# https://docs.scipy.org/doc/numpy/reference/

# torch.[ones|rand|zeroes](rows,cols)

# Consider adding dtype as an arg to newmat()
#    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
def newmat(rows, cols, init):
    a = torch.zeros(rows,cols)
    a.add_(init)
    return a

a = newmat(5,3,17).numpy()
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b) # NOTE: changing the np array changed the torch Tensor automatically


