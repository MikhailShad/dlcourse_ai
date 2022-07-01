import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.network = [
            FullyConnectedLayer(n_input, hidden_layer_size, "FL1"),
            ReLULayer("RL1"),
            FullyConnectedLayer(hidden_layer_size, n_output, "FL2")
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param_name, params in self.params().items():
            params.reset_gradients()
            assert np.all(np.isclose(params.grad, 0.))

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        input_x = X
        for layer in self.network:
            input_x = layer.forward(input_x)

        loss, gradient = softmax_with_cross_entropy(input_x, y)

        for layer in reversed(self.network):
            gradient = layer.backward(gradient)

        # After that, implement l2 regularization on all params
        for param_name, params in self.params().items():
            l2_loss, l2_gradient = l2_regularization(params.value, self.reg)
            loss += l2_loss
            params.grad += l2_gradient

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        input_x = X
        for layer in self.network:
            input_x = layer.forward(input_x)

        return np.argmax(input_x, axis=1)

    def params(self):
        result = {}

        for layer in self.network:
            params = layer.params()
            if not params:
                continue

            result[f"{layer.name}_W"] = params['W']
            result[f"{layer.name}_B"] = params['B']

        return result
