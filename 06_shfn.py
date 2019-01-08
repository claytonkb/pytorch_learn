import torch
import numpy as np

#single hidden-layer feed-forward neural net in pytorch

#http://neuralnetworksanddeeplearning.com/chap2.html
#https://github.com/claytonkb/nielsen_visuals

#data-loading:
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#

print("shfn")

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
print(xy)

# FF layer eqns:
#   a_l = act(z_l)
#   act() def= sigma()
#   z_l = (W_l * a_l-1) + b_l
#   
# BP layer eqns:
#   delta_l = grad_l (*) pd_act(z_l)
#   grad_l-1 = W_l^T * delta_l

# define nn_layer
# state:
#   input_size (n)
#   output_size (m)
#   weights (mxn matrix W)
#   biases (m-vector b)
#   results (m-vector z)
#   outputs (m-vector z)
#   deltas (m-vector d)
#   grads (n-vector g)
# methods:
#   fwd_propagate(n-vector inputs)
#   bwd_propagate(m-vector bwd_grads)

# define shfn(l,m,n)
# state:
#   nn_layer hidden (lxm layer)
#   nn_layer output (mxn layer)
#   n-vector grads
# methods:
#   fwd_propagate(l-vector inputs)
#   bwd_propagate()

# MAIN:
# open data-file
# load data-file ==> [training_set, test_set]
# train neural net
#   for 1..num_epochs:
#       shuffle(training_indices)
#       for each index (training_indices):
#           nn.fwd_propagate(trainig_set[index])
#           nn.bwd_propagate()
# calculate test-error and print
# prompt-loop:
#   input test sample # from user
#   look up sample
#   nn.fwd_propagate(sample)
#   calculate error and print

