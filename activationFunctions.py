import numpy as np 

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    exp_z = np.exp(z)
    sum = exp_z.sum()
    softmax_z = np.round(exp_z/sum, 3)
    return softmax_z

def sigmoid(x):
    exp_x = np.exp(x)
    return np.divide(exp_x, (1 + exp_x))

def tanh(x):
    return np.tanh(x)