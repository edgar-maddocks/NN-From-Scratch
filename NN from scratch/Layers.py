import numpy as np
from scipy import signal

class Layer():
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propogation(self, input):
        raise NotImplementedError("Not implemented")

    def backward_propogation(self, output_error_gradient, learning_rate):
        raise NotImplementedError("Not implemented")

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((n_neurons, 1))

    def forward_propogation(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output
    
    def backward_propogation(self, output_error_gradient, learning_rate):
        input_error_gradient = np.dot(self.weights.T, output_error_gradient)
        
        weights_error_gradient = np.dot(output_error_gradient, self.input.T)
        baises_error_gradient = output_error_gradient

        self.weights -= learning_rate * weights_error_gradient
        self.biases -= learning_rate * baises_error_gradient

        return input_error_gradient

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, n_kernels):
        input_depth, input_height, input_width = input_shape
        self.depth = n_kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (n_kernels, input_height - kernel_size + 1, input_height - kernel_size + 1)
        # 4D as it is a list of 3D kernels
        self.kernels_shape = (n_kernels, input_depth, kernel_size, kernel_size)
        #Initialize kernels + biases
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward_propogation(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward_propogation(self, output_error_gradient, learning_rate):
        input_error_gradient = np.zeros(self.input_shape)

        kernels_error_gradient = np.zeros(self.kernels_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_error_gradient[i, j] = signal.correlate2d(self.input[j], output_error_gradient[i], "valid")
                input_error_gradient[j] = signal.convolve2d(output_error_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_error_gradient
        self.biases -= learning_rate * output_error_gradient

        return input_error_gradient
        

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propogation(self, input):
        output = np.reshape(input, self.output_shape)
        return output

    def backward_propogation(self, output_error_gradient, learning_rate):
        output = np.reshape(output_error_gradient, self.input_shape)
        return output