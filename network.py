import numpy as np
import math

class activation(object):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu():
        x = np.random.random((5000, 5000)) - 0.5
        out = x * (x > 0)
        return out

class parameters:
    def __init__(self, inputs:np.array, outputs:np.array, hidden_layers:list):
        self.input_weights = inputs[0] #np.array
        self.input_biases = inputs[1] #np.array
        self.hidden_weights = hidden_layers[0] #np.array[np.array]
        self.hidden_biases = hidden_layers[1] #np.array[np.array]
        self.output_weights = outputs[0] #np.array
        self.output_biases = outputs[1] #np.array

    @staticmethod
    def generate_weights(size:list, mode:str) -> np.array:
        '''Generate a np.list containing weights with the specified size inicialized based on mode specified'''
        if mode == 'random':
            return np.random.randn(size)
        if mode == 'zero'
            return np.zeros(1)
        if mode == 'relu':
            return activation.relu():
        if mode == 'he-et-al':
            return np.random.radn(size, size-1)*math.sqrt((2/size-1))
        if mode == 'tanh' or 'glorot' or 'xavier':
            return np.random.radn(size, size-1)*math.sqrt((1/size-1))
        if mode == 'heuristic':
            return np.random.radn(size, size-1)*math.sqrt((2/(size-1+size)))
    
    @staticmethod
    def generate_biases(size:int) -> np.array:
        '''Generate a np.list containing random biases with the specified size'''
        return np.random.randn(size)

    @staticmethod
    def create_parameters(tuple:weight_shape, int:biases_size, int:hidden_column_size) -> parameters:
        # Generate weights and biases
        inputs = parameters.create_layer(weight_shape)
        outputs = Parameters.create_layer(biases_size)
        hidden_layers = parameters.create_hidden_column(hidden_column_size)

        return parameters(inputs, outputs, hidden_layers)

    def create_layer(int:layer_size) -> list[np.array,np.array]:
        '''Create random weights and biases values based on layer size'''
        weights = Parameters.generate_weights(layer_size)
        biases = Parameters.generate_biases(layer_size)
        return [weights, biases]

    def create_hidden_weights(n_layers:int) -> list:
        '''Create multiple hidden columns'''
        hidden_weights = []
        for i in range(n_layers):
            values = parameters.create_layer
            hidden_weights.append(values[0])
        return hidden_weights

    def create_hidden_biases(n_layers:int) -> list:
        '''Create multiple hidden columns'''
        hidden_biases = []
        for i in range(n_layers):
            values = parameters.create_layer
            hidden_biases.append(values[1])
        return hidden_biases

    def mutate(self, mutation_rate, mutation_chance) -> None:
        '''Change self weights ands biases'''
        def apply_mutation(array):
            for i in np.ndindex(array.shape):
                if np.random.rand() < mutation_chance:
                    array[i] += np.random.randn() * mutation_rate

        for arrays in 

        apply_mutation(self.input_weights)
        apply_mutation(self.input_biases)
        apply_mutation(self.hidden_weights)
        apply_mutation(self.hidden_biases)
        apply_mutation(self.output_weights)
        apply_mutation(self.output_biases)

    @staticmethod
    def extract_parameters(neural_network) -> parameters:
        '''Extract all weights and biases of all layers in a parameters object'''
        return parameters(neural_network.input_weights, neural_network.input_biases,
                          neural_network.hidden_weights, neural_network.hidden_biases,
                          neural_network.output_weights, neural_network.output_biases)


    def mutate_parameters(network, mutation_rate, mutation_chance):
        extracted_parameters = Parameters.extract_parameters(network)
        Parameters.mutate(extracted_parameters, mutation_rate, mutation_chance)
        return extracted_parameters
    
# Define neural network class
class RandomNeuralNetwork:
    def __init__(self, tuple:weight_shape, int:biases_size, int:hidden_column_size):
        '''Initialize random weights and biases for input layer, hidden layer, and output layer'''
        random_parameters = parameters.create_parameters(weight_shape, biases_size, hidden_column_size)
        self.input_weights = random_parameters.input_weights
        self.input_biases = random_parameters.input_biases
        self.hidden_weights = random_parameters.hidden_weights
        self.hidden_biases = random_parameters.hidden_biases
        self.output_weights = random_parameters.output_weights
        self.output_biases = random_parameters.output_biases

    def forward(self, inputs):
        '''Forward pass through the network'''
        hidden_layer_output = sigmoid(np.dot(inputs, self.input_weights) + self.input_biases)
        another_hidden_layer_output = sigmoid(np.dot(hidden_layer_output, self.hidden_weights) + self.hidden_biases)
        output = np.dot(another_hidden_layer_output, self.output_weights) + self.output_biases
        return output
    

class CustomNeuralNetwork:
    def __init__(self, parameters):
        # Set weights and biases provided as input
        self.input_weights = parameters.input_weights
        self.input_biases = parameters.input_biases
        self.hidden_weights = parameters.hidden_weights
        self.hidden_biases = parameters.hidden_biases
        self.output_weights = parameters.output_weights
        self.output_biases = parameters.output_biases

    def forward(self, inputs):
        # Forward pass through the network
        hidden_layer_output = sigmoid(np.dot(inputs, self.input_weights) + self.input_biases)
        another_hidden_layer_output = sigmoid(np.dot(hidden_layer_output, self.hidden_weights) + self.hidden_biases)
        output = np.dot(another_hidden_layer_output, self.output_weights) + self.output_biases
        return output

class generation:
    def createInitialGeneration(generation_size):
        '''Generate a list of random neural networks based on generation size parameter'''
        neural_networks = [RandomNeuralNetwork() for _ in range(generation_size)]
        return neural_networks

class Evolution:

    @staticmethod
    def select_top_half(networks_fitness):
        # Sort networks based on fitness (higher fitness values are better)
        sorted_networks = sorted(networks_fitness, key=lambda x: x[1], reverse=True)
        # Select the top half of networks
        top_half = sorted_networks[:len(sorted_networks)//2]
        return top_half

class genetics:
    def create_network_array(custom_network, fitness, generation_array:list):
        customNetworkArray = [custom_network, fitness]
        generation_array.append(customNetworkArray)

    def mutate_network_parameters(generation_array):
        mutate_custom_network_array = []
        for network, fitness in generation_array:
            mutated_parameters = Parameters.mutate_parameters(network, 0.1, 0.5)
            custom_network_mutated = CustomNeuralNetwork(mutated_parameters)
            mutate_custom_network_array.append(custom_network_mutated)
        return mutate_custom_network_array

class debugging:
    def print_network_parameters(network):
        print("Input Weights:\n", network.input_weights)
        print("Input Biases:\n", network.input_biases)
        print("Hidden Weights:\n", network.hidden_weights)
        print("Hidden Biases:\n", network.hidden_biases)
        print("Output Weights:\n", network.output_weights)
        print("Output Biases:\n", network.output_biases)
