import numpy as np
from copy import deepcopy
from .optimizer import Optimizer

# Inherits "Optimizer" for general methods and members
class SGD(Optimizer):

    def __init__(self, lr = 0.001, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum

    def set_layers(self, layers):
        super().set_layers(layers)
        self.update_previous()
    
    def update_previous(self):
        self.prev_gradient_weight = deepcopy(self.gradient_weight)
        self.prev_gradient_bias = deepcopy(self.gradient_bias)
    
    def update_params(self, x, y):

        # finds gradients (gradient_weight, gradient_bias)
        self.calculate_gradients(x, y)
        
        # applying learning rate and adding the momentum (previous gradients)
        for i in range(len(self.layers)):
            self.gradient_weight[i] = self.lr * self.gradient_weight[i] + self.momentum * self.prev_gradient_weight[i]
            self.gradient_bias[i] = self.lr * self.gradient_bias[i] + self.momentum * self.prev_gradient_bias[i]
        self.update_previous()
        
        # updating the parameters (weights, biases)
        for i in range(len(self.layers)):
            self.layers[i].weight -= self.gradient_weight[i]
            self.layers[i].bias -= self.gradient_bias[i]