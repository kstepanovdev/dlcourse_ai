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
        input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        relu_layer = ReLULayer()
        hidden_layer = FullyConnectedLayer(hidden_layer_size, n_output)

        self.layers = [input_layer, relu_layer, hidden_layer]


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
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")
        for param, value in self.params().items():
            value.grad = np.zeros_like(value.grad)

        preds = X
        for layer in self.layers:
            preds = layer.forward(preds)

        loss, grad = softmax_with_cross_entropy(preds, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        for param_name, param in self.params().items():
            if 'W' in param_name:
                l2_loss, l2_grad = l2_regularization(param.value, self.reg)

                param.grad += l2_grad
                loss += l2_loss

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        output = X

        for layer in self.layers:
            output = layer.forward(output)
        pred = np.argmax(output, axis=1)

        return pred

    def params(self):
        result = {}

        for index, layer in enumerate(self.layers):
            for param, value in layer.params().items():
                result[str(index) + param] = value

        return result
