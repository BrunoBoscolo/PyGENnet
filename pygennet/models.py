import numpy as np
import math

class activation:
    """
    Contains activation functions for the neural network.
    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

class NeuralNetwork:
    """
    A simple feedforward neural network.
    """
    def __init__(self, architecture: list[int]):
        """
        Initializes the neural network.
        'architecture' is a list of integers, where each integer represents the number of neurons in a layer.
        Example: [2, 3, 1] means 2 input neurons, 3 hidden neurons, and 1 output neuron.
        """
        if len(architecture) < 2:
            raise ValueError("Architecture must have at least an input and an output layer.")

        self.architecture = architecture

        # He-et-al initialization for weights, zeros for biases
        self.weights = []
        self.biases = []
        for i in range(len(architecture) - 1):
            # Weights for the connection between layer i and layer i+1
            weight_matrix = np.random.randn(architecture[i], architecture[i+1]) * np.sqrt(2. / architecture[i])
            self.weights.append(weight_matrix)

            # Biases for layer i+1
            bias_vector = np.zeros((1, architecture[i+1]))
            self.biases.append(bias_vector)

    def forward(self, inputs: np.array) -> np.array:
        """
        Performs a forward pass through the network.
        'inputs' should be a numpy array with shape (1, number_of_input_neurons).
        """
        if inputs.shape[1] != self.architecture[0]:
            raise ValueError(f"Input shape {inputs.shape} does not match input layer size {self.architecture[0]}.")

        current_output = inputs
        for i, (weight_matrix, bias_vector) in enumerate(zip(self.weights, self.biases)):
            current_output = np.dot(current_output, weight_matrix) + bias_vector
            # Apply activation function (e.g., sigmoid) for all but the last layer
            if i < len(self.weights) - 1:
                current_output = activation.sigmoid(current_output)

        return current_output

    def get_parameters(self) -> dict:
        """Returns the network's parameters."""
        return {
            "weights": self.weights,
            "biases": self.biases
        }

    def set_parameters(self, params: dict):
        """Sets the network's parameters from a dictionary."""
        if "weights" in params and "biases" in params:
            # Basic check for compatibility
            if len(params["weights"]) == len(self.weights) and len(params["biases"]) == len(self.biases):
                self.weights = params["weights"]
                self.biases = params["biases"]
            else:
                raise ValueError("Provided parameters do not match the network architecture.")
        else:
            raise KeyError("Parameters must contain 'weights' and 'biases' keys.")

    def mutate(self, mutation_rate: float, mutation_chance: float):
        """
        Applies mutations to the network's weights and biases.
        """
        new_weights = []
        for weight_matrix in self.weights:
            mutation_mask = np.random.random(weight_matrix.shape) < mutation_chance
            random_values = np.random.randn(*weight_matrix.shape) * mutation_rate
            new_matrix = weight_matrix + (mutation_mask * random_values)
            new_weights.append(new_matrix)
        self.weights = new_weights

        new_biases = []
        for bias_vector in self.biases:
            mutation_mask = np.random.random(bias_vector.shape) < mutation_chance
            random_values = np.random.randn(*bias_vector.shape) * mutation_rate
            new_vector = bias_vector + (mutation_mask * random_values)
            new_biases.append(new_vector)
        self.biases = new_biases


class debugging:
    """
    Utilities for inspecting network state.
    """
    @staticmethod
    def print_network_parameters(network: NeuralNetwork):
        print("Network Architecture:", network.architecture)
        print("-" * 20)
        for i, (weights, biases) in enumerate(zip(network.weights, network.biases)):
            print(f"Layer {i} -> Layer {i+1}")
            print(f"  Weights shape: {weights.shape}")
            print(f"  Biases shape:  {biases.shape}")
            # print("  Weights:\n", weights)
            # print("  Biases:\n", biases)
        print("-" * 20)
