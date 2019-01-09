import torch
import numpy as np
import math

# single hidden-layer feed-forward neural net

#http://neuralnetworksanddeeplearning.com/chap2.html
#https://github.com/claytonkb/nielsen_visuals

#data-loading:
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#

#xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
raw_data = np.loadtxt('semeion.data', delimiter=',', dtype=np.float32)

train_x = torch.from_numpy(raw_data[0:1195, 0:-10])
train_y = torch.from_numpy(raw_data[0:1195, -10:])

test_x = torch.from_numpy(raw_data[1195:, 0:-10])
test_y = torch.from_numpy(raw_data[1195:, -10:])

train_indices = np.arange(0,1195)
test_indices  = np.arange(0,397)

np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

print(test_indices[:10])

# FF layer eqns:
#   a_l = act(z_l)
#   act() def= sigma()
#   z_l = (W_l * a_l-1) + b_l
#   
# BP layer eqns:
#   delta_l = grad_l (*) pd_act(z_l)
#   grad_l-1 = W_l^T * delta_l

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def pd_sigmoid(x):
    return (x * (1 - x))

def logit(x):
    return math.log(x / (1 - x))

# define nn_layer
# state:
#   input_size (n)
#   output_size (m)
#   weights (mxn matrix W)
#   biases (m-vector b)
#   results (m-vector z)
#   outputs (m-vector a)
#   deltas (m-vector d)
#   grads (n-vector g)
# methods:
#   fwd_propagate(n-vector inputs)
#   bwd_propagate(m-vector bwd_grads)
class nn_layer:

    def __init__(self, input_size, output_size):
        self.n = input_size
        self.m = output_size
        self.x = torch.zeros(input_size,1)
        self.W = torch.rand(output_size, input_size)
        self.b = torch.rand(output_size,1)
        self.z = torch.zeros(output_size,1)
        self.a = torch.zeros(output_size,1)
        self.d = torch.zeros(output_size,1)
        self.g = torch.zeros(input_size,1)

        self.W.apply_(logit)
        self.b.apply_(logit)

    def fwd_propagate(self, inputs):
        self.x = inputs.clone()
        self.z = torch.matmul(self.W, self.x) + self.b
        self.a = self.z.clone()
        self.a.apply_(sigmoid)

    def bwd_propagate(self, grad):
        # calculate d = sigma'(z) (.) grad
        # calculate g = matmul( W^T, d )
        # calculate W = W - matmul(d, x)
        self.d = self.z.clone()
        self.d.apply_(pd_sigmoid)
        self.d = self.d * grad # Hadamard product
        self.g = torch.matmul( self.W.transpose(0,1), self.d )
        self.W = self.W - torch.matmul(self.d, self.x.transpose(0,1))

#my_layer = nn_layer(10,10)
#temp = torch.rand(10,1)
#temp.apply_(logit)
#print(temp)
#my_layer.fwd_propagate(temp)
#print(my_layer.z)
#print(my_layer.a)

#temp = torch.rand(10,1)
#print(temp)

# define shfn(l,m,n)
# state:
#   nn_layer hidden (lxm layer)
#   nn_layer output (mxn layer)
#   n-vector grads
# methods:
#   fwd_propagate(l-vector inputs)
#   bwd_propagate()
class shfn:

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden = nn_layer(input_size, hidden_size)
        self.output = nn_layer(hidden_size, output_size)
        self.loss_grad = torch.zeros(output_size,1)

    def train(self, inputs, outputs):
        self.fwd_propagate(inputs)
        self.loss_grad = (outputs - self.output.a) # gradient of MSE loss
        self.bwd_propagate()

    def fwd_propagate(self, inputs):
        self.hidden.fwd_propagate(inputs)
        self.output.fwd_propagate(self.hidden.a)

    def bwd_propagate(self):
        self.output.bwd_propagate(self.loss_grad)
        self.hidden.bwd_propagate(self.output.g)

my_shfn = shfn(7,5,3)
fake_inputs  = torch.rand(7,1)
fake_outputs = torch.rand(3,1)

print(my_shfn.hidden.W)
my_shfn.train(fake_inputs, fake_outputs) 
print(my_shfn.hidden.W)

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

