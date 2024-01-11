import numpy as np
from Model import Network
from Layers import Convolutional, Dense, Reshape
import Losses
from Activations import Activation, tanh, tanh_derivative, sigmoid, sigmoid_derivative

from keras.datasets import mnist
from keras.utils import to_categorical


from numba import jit, cuda
from timeit import default_timer as timer

def prepare_data(x_train, y_train, x_test, y_test, train_size = 0, test_size = 0):
    if train_size > x_train.shape[0]:
        print(f"train size greater than size of dataset ({x_train.shape[0]})")
        train_size = x_train.shape[0]
    elif train_size <= 0:
        print(f"train ize less than or equal to 0")
        train_size = 1000

    if test_size > x_test.shape[0]:
        print(f"train size greater than size of dataset ({x_test.shape[0]})")
        test_size = x_test.shape[0]
    elif test_size <= 0:
        print(f"train ize less than or equal to 0")
        test_size = 1000

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_train = y_train.reshape(len(y_train), 10, 1,)
    y_test = y_test.reshape(len(y_test), 10, 1,)

    return x_train, y_train, x_test, y_test

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test, train_size = x_train.shape[0], test_size = 1000)

network = Network([
    Convolutional((1, 28, 28), 3, 6), 
    Activation(sigmoid, sigmoid_derivative),
    Reshape((6, 26, 26), (6 * 26 * 26, 1)),
    Dense(6 * 26 * 26, 100),
    Activation(sigmoid, sigmoid_derivative),
    Dense(100, 10),
    Activation(sigmoid, sigmoid_derivative)
    ])

network.use(Losses.categorical_cross_entropy, Losses.categorical_cross_entropy_derivative)

network.fit(x_train, y_train, epochs = 50, learning_rate= 0.01)

result = network.predict(x_test)

for i in range(len(result)):
    print(f"Prediction:", {np.argmax(result[i])}, "True Value:", {np.argmax(y_test[i])})

print(network.accuracy(result, y_test) * 100)

