import numpy as np
from random import randrange
import copy


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def dsigmoid(x):
    y = np.multiply(x, (1-x))
    return y


def tanh(x):
    y = (2 / (1 + np.exp(-2*x))) - 1
    return y


def dtanh(x):
    y = 1 - np.power(tanh(x),2)
    return y


def mutate(x):
    if np.random.random() < 0.1:
        offset = np.random.normal() * 0.5
        return x + offset
    else:
        return x


class Perceptron:
    def __init__(self, input_count, learning_rate):
        self.input_count = input_count
        self.learning_rate = learning_rate
        self.weights = []
        for i in range(input_count):
            self.weights.append(randrange(-1, 1, 0.01))

    def guess(self, inputs: list):
        total = 0.0
        for i in range(self.input_count):
            total += self.weights[i] * inputs[i]
        return self.activate(total)

    def train(self, inputs: list, expected):
        guess_output = self.guess(inputs)
        error = expected - guess_output
        for i in range(self.input_count):
            self.weights[i] += self.learning_rate * error * inputs[i]

    @staticmethod
    def activate(total):
        if total > 0:
            return 1
        else:
            return 0


class NeuralNetwork:
    def __init__(self, input_count: int, hidden_node_counts: list, output_count: int, learning_rate=0.1,
                 activator=sigmoid, dactivator=dsigmoid):
        self.input_count = input_count
        self.hidden_node_counts = hidden_node_counts
        self.output_count = output_count
        self.learning_rate = learning_rate
        self.activator = activator
        self.dactivator = dactivator
        self.hidden_layers = []
        last_count = input_count
        for layer_node in hidden_node_counts:
            new_layer = np.random.normal(size=(layer_node, last_count))
            self.hidden_layers.append(new_layer)
            last_count = layer_node
        self.weight_output = np.random.normal(size=(self.output_count, last_count))

    def copy(self):
        my_copy = NeuralNetwork(self.input_count, self.hidden_node_counts,
                                self.output_count, self.learning_rate, self.activator, self.dactivator)
        my_copy.hidden_layers = copy.deepcopy(self.hidden_layers)
        my_copy.weight_output = self.weight_output.copy()
        return my_copy

    def mutate(self):
        new_hidden_layers = []
        for layer in self.hidden_layers:
            new_layer = mutate(layer)
            new_hidden_layers.append(new_layer)
        self.hidden_layers = new_hidden_layers
        self.weight_output = mutate(self.weight_output)

    def guess(self, inputs_list: list):
        inputs = np.matrix(inputs_list).transpose()

        layer_input = inputs.copy()

        for layer in self.hidden_layers:
            layer_input = layer.dot(layer_input)
        hidden_outputs = self.activator(layer_input)

        output_inputs = self.weight_output.dot(hidden_outputs)

        outputs = self.activator(output_inputs)
        return outputs.tolist()

    def train(self, inputs_list: list, targets_list: list):
        inputs = np.matrix(inputs_list).transpose()
        targets = np.matrix(targets_list).transpose()

        layer_inputs = [inputs]

        for layer in self.hidden_layers:
            layer_inputs.append(self.activator(layer.dot(layer_inputs[-1])))

        output_inputs = self.weight_output.dot(layer_inputs[-1])

        outputs = self.activator(output_inputs)

        output_errors = targets - outputs

        weight_output_transpose = self.weight_output.transpose()
        gradient_output = self.dactivator(outputs)

        gradient_output = np.multiply(gradient_output, output_errors)
        gradient_output = np.multiply(gradient_output, self.learning_rate)

        hidden_errors = weight_output_transpose.dot(output_errors)

        for ind in range(len(self.hidden_layers)-1, -1, -1):

            gradient_layer = self.dactivator(layer_inputs[ind+1])
            gradient_layer = np.multiply(gradient_layer, hidden_errors)
            gradient_layer = np.multiply(gradient_layer, self.learning_rate)
            layer_transpose = self.hidden_layers[ind].transpose()
            hidden_errors = layer_transpose.dot(layer_inputs[ind+1])
            layer_input_transpose = layer_inputs[ind].transpose()
            delta_w_layer = gradient_layer.dot(layer_input_transpose)
            self.hidden_layers[ind] += delta_w_layer

        hidden_outputs_transpose = layer_inputs[-1].transpose()
        delta_w_output = gradient_output.dot(hidden_outputs_transpose)
        self.weight_output += delta_w_output
