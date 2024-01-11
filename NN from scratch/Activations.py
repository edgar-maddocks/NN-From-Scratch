from Layers import Layer

import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_propogation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propogation(self, output_error_gradient, learning_rate):
        return np.multiply(output_error_gradient, self.activation_derivative(self.input))
    

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))