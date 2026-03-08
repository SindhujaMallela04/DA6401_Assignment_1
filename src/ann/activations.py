"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
    
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigma_x = sigmoid(x)
    return sigma_x * (1 - sigma_x)

def tanh(x) : 
    return np.tanh(x)

def tanh_derivative(x) :
    return 1 - np.tanh(x) ** 2

def softmax(x) : 
    exp_x = np.exp(x - np.max(x, axis = -1, keepdims = True))
    exp_x_sum = np.sum(exp_x, axis = -1, keepdims = True)
    probablities = exp_x / exp_x_sum
    return probablities
