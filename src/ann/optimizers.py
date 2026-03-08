"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD :
    def __init__(self, learning_rate = 0.01) :
        self.learning_rate = learning_rate

    def update(self, layer) :
        layer.W -= self.learning_rate * layer.grad_W
        layer.b -= self.learning_rate * layer.grad_b

class Momentum :
    def __init__(self, learning_rate = 0.01, momentum = 0.9) :
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = {}
        self.velocity_b = {}

    def update(self, layer) :
        layer_id = id(layer)

        if layer_id not in self.velocity_W :
            self.velocity_W[layer_id] = np.zeros_like(layer.grad_W)
            self.velocity_b[layer_id] = np.zeros_like(layer.grad_b)

        self.velocity_W[layer_id] = (
            self.momentum * self.velocity_W[layer_id] + (1 - self.momentum) * layer.grad_W
        )
        self.velocity_b[layer_id] = (
            self.momentum * self.velocity_b[layer_id] + (1 - self.momentum) * layer.grad_b
        )

        layer.W -= self.learning_rate * self.velocity_W[layer_id]
        layer.b -= self.learning_rate * self.velocity_b[layer_id]

class NAG :
    def __init__(self, learning_rate = 0.001, momentum = 0.9) :
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = {}
        self.velocity_b = {}

    def update(self, layer) :
        layer_id = id(layer)

        if layer_id not in self.velocity_W :
            self.velocity_W[layer_id] = np.zeros_like(layer.W)
            self.velocity_b[layer_id] = np.zeros_like(layer.b)

        prev_v_W = self.velocity_W[layer_id]
        prev_v_b = self.velocity_b[layer_id]

        self.velocity_W[layer_id] = self.momentum * prev_v_W + layer.grad_W
        self.velocity_b[layer_id] = self.momentum * prev_v_b + layer.grad_b

        layer.W -= self.learning_rate * (self.momentum * self.velocity_W[layer_id] + layer.grad_W)
        layer.b -= self.learning_rate * (self.momentum * self.velocity_b[layer_id] + layer.grad_b)

class RMSProp :
    def __init__(self, learning_rate = 0.001, beta = 0.9, epsilon = 1e-8) :
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.run_avg_sqd_grad_W = {}
        self.run_avg_sqd_grad_b = {}

    def update(self, layer) :
        layer_id = id(layer)

        if layer_id not in self.run_avg_sqd_grad_W :
            self.run_avg_sqd_grad_W[layer_id] = np.zeros_like(layer.grad_W)
            self.run_avg_sqd_grad_b[layer_id] = np.zeros_like(layer.grad_b)

        self.run_avg_sqd_grad_W[layer_id] = (
            self.beta * self.run_avg_sqd_grad_W[layer_id] + (1 - self.beta) * (layer.grad_W ** 2)
        )
        self.run_avg_sqd_grad_b[layer_id] = (
            self.beta * self.run_avg_sqd_grad_b[layer_id] + (1 - self.beta) * (layer.grad_b ** 2)
        )
    
        layer.W -= (self.learning_rate * layer.grad_W / (np.sqrt(self.run_avg_sqd_grad_W[layer_id]) + self.epsilon))
        layer.b -= (self.learning_rate * layer.grad_b / (np.sqrt(self.run_avg_sqd_grad_b[layer_id]) + self.epsilon))
