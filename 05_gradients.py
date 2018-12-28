import torch
import numpy as np

#https://explained.ai/matrix-calculus/index.html
#http://neuralnetworksanddeeplearning.com/chap2.html
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

print("gradients")

# FF layer eqns:
#   a_l = act(z_l)
#   act() def= sigma()
#   z_l = (W_l * a_l-1) + b_l
#   
# BP layer eqns:
#   delta_l = grad_l (*) pd_act(z_l)
#   grad_l-1 = W_l^T * delta_l



