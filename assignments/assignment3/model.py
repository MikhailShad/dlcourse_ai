import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        image_width, image_height, n_channels = input_shape
        filter_size = 3
        padding = 1
        pool_stride = 4
        pool_size = 4
        fcl_inputs = conv2_channels * image_width * image_height // (pool_stride ** 4)
        self.layers = [
            ConvolutionalLayer(n_channels, conv1_channels, filter_size, padding, name="CL1"),
            ReLULayer(name="RL2"),
            MaxPoolingLayer(pool_size, pool_stride, name="MaxPool3"),
            ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding, name="CL4"),
            ReLULayer(name="RL5"),
            MaxPoolingLayer(pool_size, pool_stride, name="MaxPool6"),
            Flattener(name="Flattener7"),
            FullyConnectedLayer(fcl_inputs, n_output_classes, "FL8")
        ]

    def forward(self, X):
        x = X.copy()
        for layer in self.layers:
            # print(f"Forward. Layer = {layer.name}")
            x = layer.forward(x)

        return x

    def backward(self, grad):
        d_out = grad.copy()
        for layer in reversed(self.layers):
            # print(f"Backward. Layer = {layer.name}")
            d_out = layer.backward(d_out)

        return d_out

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for p in self.params().values():
            p.grad = np.zeros_like(p.grad)

        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        predictions = self.forward(X)

        loss, grad = softmax_with_cross_entropy(predictions, y)

        self.backward(grad)

        return loss

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

    def params(self):
        result = {}

        for layer in self.layers:
            if not layer.params():
                continue

            for k, v in layer.params().items():
                result[f"{layer.name}_{k}"] = v

        return result
