import numpy as np

def mse(true_value, prediction):
    return np.mean(np.power(true_value - prediction, 2))

def mse_derivative(true_value, prediction):
    return 2*(prediction-true_value)/true_value.size


def categorical_cross_entropy(true_value, prediction):
    losses = []
    for t,p in zip(true_value, prediction):
      loss = -np.sum(t * np.log(p))
      losses.append(loss)
    return np.sum(losses)

def categorical_cross_entropy_derivative(true_value, prediction):
    return prediction - true_value