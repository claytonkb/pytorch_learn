import torch
import numpy as np

# Python array notation:
#
# a[start:end] # items start through end-1
# a[start:]    # items start through the rest of the array
# a[:end]      # items from the beginning through end-1
# a[:]         # a copy of the whole array
#
# There is also the step value, which can be used with any of the above:
# 
# a[start:end:step] # start through not past end, by step
#
# The key point to remember is that the :end value represents the first value
# that is not in the selected slice. So, the difference beween end and start is
# the number of elements selected (if step is 1, the default).
#
# The other feature is that start or end may be a negative number, which means it
# counts from the end of the array instead of the beginning. So:
#
# a[-1]    # last item in the array
# a[-2:]   # last two items in the array
# a[:-2]   # everything except the last two items
#
# Similarly, step may be a negative number:
# 
# a[::-1]    # all items in the array, reversed
# a[1::-1]   # the first two items, reversed
# a[:-3:-1]  # the last two items, reversed
# a[-3::-1]  # everything except the last two items, reversed

# Multi-dimensional slices:
#
# 3x2x2 tensor
a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

print("a")
print(a)
print("a[0]")
print(a[0])
print("a[:,0]")
print(a[:,0])
print("a[:,:,0]")
print(a[:,:,0])

print("a[:,0][0][1:3]")
print(a[:,0][0][1:3])

# Note: start:end syntax indexes on boundaries between elements

exit()

# torch.where(condition, x, y) --> Tensor
# This is a filter

#torch.split(tensor, split_size_or_sections, dim=0)[SOURCE]
#Splits the tensor into chunks.
#
#If split_size_or_sections is an integer type, then tensor will be split into
#equally sized chunks (if possible). Last chunk will be smaller if the tensor
#size along the given dimension dim is not divisible by split_size.

# concatenation

x = torch.randn(2, 3)
print(x)
y = torch.cat((x, x, x), 0)
print(y)
z = torch.cat((x, x, x), 1)
print(z)

# split (chunk)
# torch.chunk(tensor, chunks, dim=0) → List of Tensors
# Splits a tensor into a specific number of chunks.
#
# Last chunk will be smaller if the tensor size along the given dimension dim is
# not divisible by chunks.

# gather (axis-wise select)
#>>> t = torch.tensor([[1,2],[3,4]])
#>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
#tensor([[ 1,  1],
#        [ 4,  3]])

# index_select (multi-select)
#
#>>> x = torch.randn(3, 4)
#>>> x
#tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
#        [-0.4664,  0.2647, -0.1228, -1.1068],
#        [-1.1734, -0.6571,  0.7230, -0.6004]])
#>>> indices = torch.tensor([0, 2])
#>>> torch.index_select(x, 0, indices)
#tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
#        [-1.1734, -0.6571,  0.7230, -0.6004]])
#>>> torch.index_select(x, 1, indices)
#tensor([[ 0.1427, -0.5414],
#        [-0.4664, -0.1228],
#        [-1.1734,  0.7230]])
# Explanation:
#
# First example selects vectors 0 and 2 along dimension 0 (horizontal):
#    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
#            [ ******,  ******,  ******,  ******],
#            [-1.1734, -0.6571,  0.7230, -0.6004]])
#       Asterisks mark the items NOT selected
#
# Second example selects vectors 0 and 2 along dimension 1 (vertical):
#tensor([[ 0.1427,  ******, -0.5414,  ******],
#        [-0.4664,  ******, -0.1228,  ******],
#        [-1.1734,  ******,  0.7230,  ******]])
#       Asterisks mark the items NOT selected

# torch.reshape(input, shape) → Tensor
#
#Returns a tensor with the same data and number of elements as input, but with
#the specified shape. When possible, the returned tensor will be a view of
#input. Otherwise, it will be a copy. Contiguous inputs and inputs with
#compatible strides can be reshaped without copying, but you should not depend
#on the copying vs. viewing behavior.



