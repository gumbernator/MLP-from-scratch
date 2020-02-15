import numpy as np

class Layer:

    def __init__(self, input_num, neuron_num, activation):
        self.weight = (np.random.rand(input_num, neuron_num) - 0.5) / 10
        self.bias = (np.random.rand(1, neuron_num) - 0.5) / 10
        self.activation = activation
        self.neuron_num = neuron_num
    
    def set_data_type(self, data_type):
        self.weight = self.weight.astype(data_type)
        self.bias = self.bias.astype(data_type)
        self.data_type = data_type

    def forward(self, x):
        self.output = self.activation(x.dot(self.weight) + self.bias).astype(self.data_type)
        return self.output
    
    def deriv(self):
        return self.activation(self.output, deriv = True)