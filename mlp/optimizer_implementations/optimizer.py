import numpy as np

class Optimizer:

    def set_loss(self, loss):
        self.loss = loss
        self.batch_loss = 0

    def set_layers(self, layers):
        self.layers = layers
        self.gradient_weight = [] # weight gradient
        self.gradient_bias = [] # bias gradient
        
        self.data_type = layers[0].data_type

        # all parameter gradients are initialized to zero
        for layer in self.layers:
            self.gradient_weight.append(np.zeros(layer.weight.shape).astype(self.data_type))
            self.gradient_bias.append(np.zeros(layer.bias.shape).astype(self.data_type))

    def feedforward(self, x):
        running_layer = np.copy(x).astype(self.data_type)
        for layer in self.layers:
            running_layer = layer.forward(running_layer)
        return running_layer

    # finds gradients based on given data (x, y)
    def calculate_gradients(self, x, y):
        # getting prediction
        _y = self.feedforward(x)

        # loss to display
        self.batch_loss = np.sum(self.loss(_y, y)) / len(y)

        # the running loss
        running_loss = self.loss(_y, y, deriv = True)

        # calculating gradients from last to second layers
        for i in range(len(self.layers)-1, 0, -1):
            d_bias = running_loss * self.layers[i].deriv()
            self.gradient_bias[i] = np.sum(d_bias, axis = 0).reshape(1, -1) / len(x)
            d_weight = self.layers[i-1].output.T.dot(d_bias) / len(x)
            self.gradient_weight[i] = d_weight

            # backpropagating the running loss
            running_loss = running_loss.dot(self.layers[i].weight.T)

        # calculating gradient for a first layer
        d_bias = running_loss * self.layers[0].deriv()
        self.gradient_bias[0] = np.sum(d_bias, axis = 0).reshape(1, -1) / len(x)
        d_weight = x.T.dot(d_bias) / len(x)
        self.gradient_weight[0] = d_weight
        