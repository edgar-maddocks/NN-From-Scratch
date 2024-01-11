import numpy as np

class Network():
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.loss_derivative = None
        print("Layers initialized...")

    def use(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative
        print("Error functions initialized...")

    def predict(self, input):
        result = []
        for i in range(len(input)):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propogation(output)
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, epochs=1000, learning_rate = 0.1, batch_size = None):
        print("Model beginning training...")
        for i in range(epochs):
            total_loss = 0
            if batch_size is None:
                for j in range(len(x_train)):
                    output = x_train[j]
                    for layer in self.layers:
                        output = layer.forward_propogation(output)
                    total_loss += self.loss(y_train[j], output)

                    output_error_gradient = self.loss_derivative(y_train[j], output)
                    for layer in reversed(self.layers):
                        output_error_gradient = layer.backward_propogation(output_error_gradient, learning_rate)
            else:
                for batch in self.get_minibatches(y_train, x_train, batch_size, shuffle=True):
                    x_batch, y_batch = batch
                    batch_loss = 0
                    list_of_output_error_gradients = []
                    for j in range(len(x_batch)):
                        output = x_batch[j]
                        for layer in self.layers:
                            output = layer.forward_propogation(output)
                        batch_loss += self.loss(y_batch[j], output)

                        list_of_output_error_gradients.append(self.loss_derivative(y_batch[j], output))

                    output_error_gradient = np.mean(list_of_output_error_gradients, axis=0)
                    for layer in reversed(self.layers):
                        output_error_gradient = layer.backward_propogation(output_error_gradient, learning_rate)
                    total_loss += batch_loss
            total_loss /= len(x_train)
            

            print(f"Epoch {i+1} / {epochs}    Error={total_loss}")

    def accuracy(self, results, true_values):
        result = [np.argmax(result) for result in results]
        true_values = [np.argmax(true_value) for true_value in true_values]

        return sum(1 for x,y in zip(result, true_values) if x == y) / len(result)
    
    def get_minibatches(self, y, x, batch_size, shuffle=False):
        if shuffle:
            # creates list of indices for each data row in predictions
            indices = np.arange(len(x))
            np.random.shuffle(indices)
        for start_idx in range(0, len(x) - batch_size + 1, batch_size):
            if shuffle:
                batch = indices[start_idx:start_idx + batch_size]
            else:
                batch = slice(start_idx, start_idx + batch_size)
            yield x[batch], y[batch]