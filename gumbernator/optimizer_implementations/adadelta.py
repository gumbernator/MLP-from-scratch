import numpy as np
from .optimizer import Optimizer

##########
# FAILED #
##########

# Inherits "Optimizer" for general methods and members
class Adadelta(Optimizer):

    def __init__(self, lr = 1.0, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate

        self.avg_grad_weight = [] # decaying average of past squared weights
        self.avg_grad_bias = [] # decaying average of past squared biases

        self.last_weight_change = [] # the last weight changes squared (to substitute learning rate)
        self.last_bias_change = [] # the last bias changes squared (to substitute learning rate)

        self.epsilon = 1e-5
        self.first_update = True
    
    def set_layers(self, layers):
        super().set_layers(layers)

        for layer in self.layers:
            weight_zeros = np.zeros(((layer.weight.shape[0]), 1)).astype(self.data_type)
            self.avg_grad_weight.append(np.copy(weight_zeros))
            self.last_weight_change.append(np.copy(weight_zeros))

            bias_zeros = np.zeros(((layer.bias.shape[0]), 1)).astype(self.data_type)
            self.avg_grad_bias.append(np.copy(bias_zeros))
            self.last_bias_change.append(np.copy(bias_zeros))
    
    def update_params(self, x, y):
        
        # finds gradients (gradient_weight, gradient_bias)
        self.calculate_gradients(x, y)

        if self.first_update:
            for i in range(len(self.layers)):
                self.avg_grad_weight[i] = self.gradient_weight[i]**2
                self.avg_grad_bias[i] = self.gradient_bias[i]**2
        
        # updating the parameters (weights, biases)
        for i in range(len(self.layers)):
            weight_change = np.sqrt((self.last_weight_change[i] + self.epsilon) / (self.epsilon + self.avg_grad_weight[i])) * self.gradient_weight[i]
            self.layers[i].weight -= weight_change
            bias_change = np.sqrt((self.last_bias_change[i] + self.epsilon) / (self.epsilon + self.avg_grad_bias[i])) * self.gradient_bias[i]
            self.layers[i].bias -= bias_change
            
            # updating last parameter changes
            self.last_weight_change[i] = self.decay_rate * self.last_weight_change[i] + (1 - self.decay_rate) * np.sum(weight_change**2, axis = 1).reshape(-1,1)
            self.last_bias_change[i] = self.decay_rate * self.last_bias_change[i] + (1 - self.decay_rate) * np.sum(bias_change**2, axis = 1).reshape(-1,1)
        print (weight_change[1])

        if not self.first_update:
            # updating past average squared gradients 
            for i in range(len(self.layers)):
                self.avg_grad_weight[i] = self.decay_rate * self.avg_grad_weight[i] + (1 - self.decay_rate) * np.sum(self.gradient_weight[i]**2, axis = 1).reshape(-1,1)
                self.avg_grad_bias[i] = self.decay_rate * self.avg_grad_bias[i] + (1 - self.decay_rate) * np.sum(self.gradient_bias[i]**2, axis=1).reshape(-1,1)
        
        if self.first_update:
            self.first_update = False