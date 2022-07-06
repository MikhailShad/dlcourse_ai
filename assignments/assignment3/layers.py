import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    predictions = predictions.copy()
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
    shifted = predictions - np.max(predictions, axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents, axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape) == 1:
        probs = probs.copy().reshape(1, -1)

    batch_size = probs.shape[0]

    return -np.mean(np.log(probs[np.arange(batch_size), target_index]))


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = W * 2 * reg_strength

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    if len(predictions.shape) == 1:
        predictions = np.array([predictions])

    probs = softmax(predictions)
    batch_size = probs.shape[0]

    loss = cross_entropy_loss(probs, target_index)

    dprediction = probs.copy()
    dprediction[range(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self, name="RL"):
        self.name = name
        self.relu_result = None
        pass

    def forward(self, X):
        self.relu_result = np.maximum(X, 0)
        return self.relu_result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        # Implement backward pass
        d_result = d_out * np.sign(self.relu_result)  # dL / dx
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output, name="FL"):
        self.name = name
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = d_out.sum(0)[None, :]
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding, stride=1, name="CL"):
        """
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        """
        self.name = name
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

        # Custom fields
        self.X = None
        self.stride = stride

    def forward(self, X):
        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                        mode='constant', constant_values=0)
        batch_size, height, width, channels = X.shape

        out_height = 1 + (height - self.filter_size + 2 * self.padding) / self.stride
        assert int(out_height) == out_height  # check if hyperparameters fit the model
        out_height = int(out_height)

        out_width = 1 + (width - self.filter_size + 2 * self.padding) / self.stride
        assert int(out_width) == out_width  # check if hyperparameters fit the model
        out_width = int(out_width)

        # Implement forward pass
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_area = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :]
                x_reshaped = x_area.reshape((batch_size, self.filter_size * self.filter_size * self.in_channels))
                w_reshaped = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels,
                                                  self.out_channels)
                result[:, y, x, :] = np.dot(x_reshaped, w_reshaped) + self.B.value

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        self.B.grad = np.sum(d_out, axis=(0, 1, 2))

        grad_x = np.zeros((batch_size, height, width, channels))
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                x_area = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :]
                self.W.grad += np.transpose(np.dot(np.transpose(x_area, [3, 1, 2, 0]), d_out[:, y, x, :]), [1, 2, 0, 3])
                grad_x[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.dot(d_out[:, y, x, :],
                                                                                       np.transpose(self.W.value,
                                                                                                    [0, 1, 3, 2]))

        return grad_x[:, self.padding:height - self.padding, self.padding:width - self.padding, :]  # get rid of padding

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride, name="MaxPool"):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.name = name

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = X.shape

        out_height = 1 + (height - self.pool_size) / self.stride
        assert int(out_height) == out_height  # check if hyperparameters fit the model
        out_height = int(out_height)

        out_width = 1 + (width - self.pool_size) / self.stride
        assert int(out_width) == out_width  # check if hyperparameters fit the model
        out_width = int(out_width)

        # Implement forward pass
        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                x_area = self.X[:, y * self.stride:y * self.stride + self.pool_size,
                         x * self.stride:x * self.stride + self.pool_size, :]
                result[:, y, x, :] += np.max(x_area, axis=(1, 2))

        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_input = np.zeros_like(self.X)

        batch_idxs = np.repeat(np.arange(batch_size), channels)
        channel_idxs = np.tile(np.arange(channels), batch_size)

        for y in range(out_height):
            for x in range(out_width):
                slice_x = self.X[:, y * self.stride:y * self.stride + self.pool_size,
                          x * self.stride:x * self.stride + self.pool_size, :].reshape(batch_size, -1, channels)

                max_idxs = np.argmax(slice_x, axis=1)

                slice_d_input = d_input[:, y * self.stride:y * self.stride + self.pool_size,
                                x * self.stride:x * self.stride + self.pool_size, :].reshape(batch_size, -1, channels)

                slice_d_input[batch_idxs, max_idxs.flatten(), channel_idxs] = d_out[batch_idxs, y, x, channel_idxs]

                d_input[:, y * self.stride:y * self.stride + self.pool_size,
                x * self.stride:x * self.stride + self.pool_size, :] = slice_d_input.reshape(batch_size, self.pool_size,
                                                                                             self.pool_size,
                                                                                             channels)

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self, name="Flattener"):
        self.X_shape = None
        self.name = name

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        #  Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = batch_size, height, width, channels  # исходные размеры
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
