import numpy as np
from copy import deepcopy
from .optimizer import Optimizer

##########
# FAILED #
##########

# Inherits "Optimizer" for general methods and members
class StopAndTurn(Optimizer):

    def __init__(self, first_step = 0.1, minimum = 1e-5, divider = 2, limit = 3):
        self.first_step = first_step
        self.minimum = minimum
        self.divider = divider
        self.limit = limit

        self.step_weight = []
        self.step_bias = []

    def set_layers(self, layers):
        super().set_layers(layers)
        
        for layer in self.layers:
            weight_zeros = np.full(layer.weight.shape, self.first_step).astype(self.data_type)
            self.step_weight.append(weight_zeros)
            bias_zeros = np.full(layer.bias.shape,  self.first_step).astype(self.data_type)
            self.step_bias.append(bias_zeros)
    
    def update_params(self, x, y):

        # finds gradients (gradient_weight, gradient_bias)
        self.calculate_gradients(x, y)
        
        for i in range(len(self.layers)):
            diff_weight = np.sign(self.step_weight[i] * self.gradient_weight[i])
            self.step_weight[i][diff_weight == -1] /= self.divider
            diff_bias = np.sign(self.step_bias[i] * self.gradient_bias[i])
            self.step_bias[i][diff_bias == -1] /= self.divider

        # updating the parameters (weights, biases)
        for i in range(len(self.layers)):
            self.layers[i].weight -= self.step_weight[i] * self.gradient_weight[i]
            self.layers[i].bias -= self.step_bias[i] * self.gradient_bias[i]

            # self.layers[i].weight[self.layers[i].weight > self.limit] = self.limit
            # self.layers[i].weight[self.layers[i].weight < -self.limit] = -self.limit
            # self.layers[i].bias[self.layers[i].bias > self.limit] = self.limit
            # self.layers[i].bias[self.layers[i].bias < -self.limit] = -self.limit