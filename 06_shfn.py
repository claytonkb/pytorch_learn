import torch
import numpy as np
import math

#http://neuralnetworksanddeeplearning.com/chap2.html
#https://github.com/claytonkb/nielsen_visuals

#data-loading:
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#

raw_data = np.loadtxt('semeion.data', delimiter=',', dtype=np.float32)

temp_train_x = torch.from_numpy(raw_data[0:1195, 0:-10])
temp_train_y = torch.from_numpy(raw_data[0:1195, -10:])

temp_test_x = torch.from_numpy(raw_data[1195:, 0:-10])
temp_test_y = torch.from_numpy(raw_data[1195:, -10:])

train_x = [torch.Tensor() for _ in range(1195)]
train_y = [torch.Tensor() for _ in range(1195)]

test_x = [torch.Tensor() for _ in range(397)]
test_y = [torch.Tensor() for _ in range(397)]

for i in range(0,1195):
    train_x[i] = torch.Tensor(256,1)
    train_y[i] = torch.Tensor(10,1)
    train_x[i] = temp_train_x[i].reshape(256,1)
    train_y[i] = temp_train_y[i].reshape(10,1)

for i in range(0,397):
    test_x[i] = torch.Tensor(256,1)
    test_y[i] = torch.Tensor(10,1)
    test_x[i] = temp_test_x[i].reshape(256,1)
    test_y[i] = temp_test_y[i].reshape(10,1)

# FF layer eqns:
#   a_l = act(z_l)
#   act() def= sigma()
#   z_l = (W_l * a_l-1) + b_l
#   
# BP layer eqns:
#   delta_l = grad_l (*) pd_act(z_l)
#   grad_l-1 = W_l^T * delta_l

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.5

def pd_sigmoid(x):
    u = math.exp(x)
    return u / ((1 + u) ** 2)

def logit(x):
    return math.log(x / (1 - x))

class nn_layer:

    eta = 0.1

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
        # update    W = W - eta * matmul(d, x)
        # update    b = b - eta * d
        self.d = self.z.clone()
        self.d.apply_(pd_sigmoid)
        #self.d = self.d * grad # Hadamard product
        torch.Tensor.mul_(self.d, grad) # Hadamard product
        self.g = torch.matmul( self.W.transpose(0,1), self.d )
        self.W = self.W - self.eta * torch.matmul(self.d, self.x.transpose(0,1))
        self.b = self.b - self.eta * self.d

# Single Hidden-layer Feed-forward Neural-net
# Consists of two nn_layers (hidden, output) and a loss-gradient
class shfn:

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden = nn_layer(input_size, hidden_size)
        self.output = nn_layer(hidden_size, output_size)
        self.loss_grad = torch.ones(output_size,1)

    def train(self, inputs, outputs):
        self.fwd_propagate(inputs)
        self.loss_grad = (self.output.a - outputs) # gradient of MSE loss
        self.bwd_propagate()

    def fwd_propagate(self, inputs):
        self.hidden.fwd_propagate(inputs)
        self.output.fwd_propagate(self.hidden.a)

    def bwd_propagate(self):
        self.output.bwd_propagate(self.loss_grad)
        self.hidden.bwd_propagate(self.output.g)

train_indices = np.arange(0,1195)

my_shfn = shfn(256,32,10)

#tempW = my_shfn.hidden.W.clone()
#tempW = my_shfn.output.W.clone()

for j in range(0,10):
    print("epoch: ", j, "\n")
    np.random.shuffle(train_indices)
    for i in range(0,1195):
        my_shfn.train(train_x[train_indices[i]], train_y[train_indices[i]]) 

#print(my_shfn.hidden.W == tempW)
#print(my_shfn.output.W == tempW)

for i in range(0,397):
    my_shfn.fwd_propagate(test_x[i])
    print((my_shfn.output.a - test_y[i]).transpose(0,1))

# prompt-loop:
#   input test sample # from user
#   look up sample
#   nn.fwd_propagate(sample)
#   calculate error and print

