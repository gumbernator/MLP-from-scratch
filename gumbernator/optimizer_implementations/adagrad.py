import numpy as np
from .optimizer import Optimizer

# Inherits "Optimizer" for general methods and members
class Adagrad(Optimizer):

    def __init__(self, lr = 0.01):
        self.lr = lr

        self.Gt_weight = [] # historical squared weights
        self.Gt_bias = [] # historical squared biases

        self.epsilon = 1e-5
    
    def set_layers(self, layers):
        super().set_layers(layers)

        for layer in self.layers:
            weight_zeros = np.zeros(((layer.weight.shape[0]), 1)).astype(self.data_type)
            self.Gt_weight.append(weight_zeros)
            bias_zeros = np.zeros(((layer.bias.shape[0]), 1)).astype(self.data_type)
            self.Gt_bias.append(bias_zeros)
    
    def update_params(self, x, y):
        
        # finds gradients (gradient_weight, gradient_bias)
        self.calculate_gradients(x, y)

        # updating the parameters (weights, biases)
        for i in range(len(self.layers)):
            self.layers[i].weight -= self.lr / np.sqrt(self.epsilon + self.Gt_weight[i]) * self.gradient_weight[i]
            self.layers[i].bias -= self.lr / np.sqrt(self.epsilon + self.Gt_bias[i]) * self.gradient_bias[i]

        # summing up the historical gradients squared
        for i in range(len(self.layers)):
            self.Gt_weight[i] += np.sum(self.gradient_weight[i]**2, axis = 1).reshape(-1,1)
            self.Gt_bias[i] += np.sum(self.gradient_bias[i]**2, axis=1).reshape(-1,1)
        