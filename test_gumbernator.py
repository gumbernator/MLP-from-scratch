"""
Author: Guyugmonkh Lkhagvachuluun
Github: https://github.com/gumbernator
Project: MLP from scratch

highly inspired by keras,
Adadelta optimizer is currently broken!!!
"""

from mlp.nn import MLP
from mlp.losses import mse, mae, cross_entropy
from mlp.activations import sigmoid, tanh, relu, leaky_relu, softmax
from mlp.optimizers import SGD, Adagrad, Adadelta, RMSprop

from keras.datasets import mnist
from keras.utils import to_categorical

batch_size = 128
num_classes = 10
epochs = 10

# loading MNIST Digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# making the model
model = MLP(input_num = 784)
model.add_layer(128, leaky_relu)
model.add_layer(num_classes, softmax)
model.compile(optimizer = RMSprop(lr = 0.01), loss = cross_entropy, data_type = 'float32')

# training
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1)

# evaluating
accuracy = model.evaluate_accuracy(x_test, y_test)
print ('categorical accuracy:', accuracy)
