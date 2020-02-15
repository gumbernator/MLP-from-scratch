import numpy as np
from .optimizer import Optimizer

# Inherits "Optimizer" for general methods and members
class RMSprop(Optimizer):

    def __init__(self, lr = 0.001, decay_rate = 0.9):
        self.lr = lr
        self.decay_rate = decay_rate

        self.avg_grad_weight = [] # decaying average of past squared weights
        self.avg_grad_bias = [] # decaying average of past squared biases

        self.epsilon = 1e-5
    
    def set_layers(self, layers):
        super().set_layers(layers)

        for layer in self.layers:
            weight_zeros = np.zeros(((layer.weight.shape[0]), 1)).astype(self.data_type)
            self.avg_grad_weight.append(weight_zeros)

            bias_zeros = np.zeros(((layer.bias.shape[0]), 1)).astype(self.data_type)
            self.avg_grad_bias.append(bias_zeros)
    
    def update_params(self, x, y):
        
        # finds gradients (gradient_weight, gradient_bias)
        self.calculate_gradients(x, y)
        
        # updating the parameters (weights, biases)
        for i in range(len(self.layers)):
            self.layers[i].weight -= self.lr / np.sqrt(self.epsilon + self.avg_grad_weight[i]) * self.gradient_weight[i]
            self.layers[i].bias -= self.lr / np.sqrt(self.epsilon + self.avg_grad_bias[i]) * self.gradient_bias[i]
        
        # updating past average squared gradients 
        for i in range(len(self.layers)):
            self.avg_grad_weight[i] = self.decay_rate * self.avg_grad_weight[i] + (1 - self.decay_rate) * np.sum(self.gradient_weight[i]**2, axis = 1).reshape(-1,1)
            self.avg_grad_bias[i] = self.decay_rate * self.avg_grad_bias[i] + (1 - self.decay_rate) * np.sum(self.gradient_bias[i]**2, axis=1).reshape(-1,1)