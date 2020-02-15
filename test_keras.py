"""
Source: https://keras.io/examples/mnist_mlp/
Project: MLP by keras.io
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import RMSprop

batch_size = 64
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
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# making the model
model = Sequential()
model.add(Dense(128, activation=LeakyReLU(alpha = 0.1), input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr = 0.01),
              metrics=['accuracy'])

# tarining
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# evaluating
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
