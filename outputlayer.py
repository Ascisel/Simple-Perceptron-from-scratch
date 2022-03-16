from layer import Layer
import numpy as np
from tanh import tanh_prime, tanh


class OutputLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.weights = np.zeros((input_size, output_size))
        self.biases = np.random.randn(1, output_size)

    def forward(self, input):
        self.input = input
        # self.output = SUM (x_i * theta_i + theta_n+1), from i = 1 to n
        self.output = np.dot(input, self.weights) + self.biases

        return self.output

    def output_prime(self, x):
        return 1

    """
    output_error = (dq / dy_2k)

    """
    def backward(self, output_error, learning_rate):
        # (d q / d y_1j) = (d q / d y_2k) * (dabba y_2k / dabba y_1j)
        #            = (d q / d y_2k) * theta''_k,j
        error = self.output_prime(self.output) * output_error
        self.input_error = np.dot(error, self.weights.T)

        # (d q / d theta''_k,j) = (d q / d y_2k) * (dabba y_2k / dabba theta''_k,j)
        #                       = (d q / d y_2k) * y_1j
        self.weights_error = np.dot(self.input.T, error)

        self.weights -= learning_rate * self.weights_error
        self.biases -= learning_rate * error

        return self.input_error
