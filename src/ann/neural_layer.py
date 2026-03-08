"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer :
    def __init__(self, input_size, output_size, weight_init) :
        if weight_init == 'xavier' :
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
            self.b = np.zeros(output_size)

        elif weight_init == 'random' :
            self.W = np.random.randn(input_size, output_size)
            self.b = np.zeros(output_size)

        elif weight_init == 'zeros' :
            self.W = np.zeros((input_size, output_size))
            self.b = np.zeros(output_size)

        else :
            raise ValueError("Invalid weight initialization method")
        
        self.grad_W = None
        self.grad_b = None

    def forward(self, input_data) :
        self.input = input_data
        self.Z = np.dot(input_data, self.W) + self.b
        return self.Z
    
    def backward(self, output_grad) :
        # m = self.input.shape[0]
        self.grad_W = np.dot(self.input.T, output_grad)
        self.grad_b = np.sum(output_grad, axis = 0)
        return np.dot(output_grad, self.W.T)