import numpy as np
from datetime import datetime
from .data_controller import DataController
from .layer import Layer

class MLP:

    # initializing with given input dimensions
    def __init__(self, input_num):
        self.layers = []
        self.input_num = input_num
    
    def add_layer(self, neuron_num, activation):
        if len(self.layers) > 0:
            layer = Layer(self.layers[-1].neuron_num, neuron_num, activation)
        else:
            layer = Layer(self.input_num, neuron_num, activation)
        self.layers.append(layer)
    
    # setting optimizer, loss and data_type of parameters
    def compile(self, optimizer, loss, data_type = 'float32'):
        self.data_type = data_type

        # we will use data controller to provide batches
        self.data_controller = DataController(data_type)
        for i in range(len(self.layers)):
            self.layers[i].set_data_type(data_type)

        self.optimizer = optimizer
        self.optimizer.set_layers(self.layers)
        self.optimizer.set_loss(loss)
    
    def predict(self, x):
        return self.optimizer.feedforward(x)
    
    def evaluate_accuracy(self, x, y):
        _y = self.optimizer.feedforward(x)
        _y = _y.argmax(axis = 1)
        y = y.argmax(axis = 1)
        score = np.zeros_like(y)
        score[y == _y] += 1
        return np.mean(score)
    
    def fit(self, x, y, batch_size, epochs, verbose = 1):

        # providing data to data controller
        self.data_controller.set(x, y, batch_size)

        for i in range(epochs):
            start_time = datetime.now()
            for ii, (batch_x, batch_y) in enumerate(self.data_controller.get_batches()):
                self.optimizer.update_params(batch_x, batch_y)

                if verbose == 1:
                    self.printProgressBar(
                        round(ii * batch_size * 100 / len(x)),
                        100,
                        prefix = 'epoch: {}'.format(i+1),
                        suffix = ', {}, loss: {}'.format(datetime.now() - start_time, self.optimizer.batch_loss)
                    )
            if verbose == 1: print ()
    
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 30, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
