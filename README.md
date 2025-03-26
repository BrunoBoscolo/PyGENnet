# Neural Network Evolution

This repository contains an implementation of a simple neural network with evolutionary mutation capabilities. The code defines classes for handling network parameters, mutation, and evolution.

## Features
- **Customizable Neural Network**: Supports different activation functions and weight initialization methods.
- **Evolutionary Algorithm**: Implements mutation and selection mechanisms.
- **Random Neural Network Generation**: Allows the creation of randomized networks for evolutionary training.
- **Debugging Tools**: Includes utilities to inspect network parameters.

## Installation
Ensure you have Python and NumPy installed:
```bash
pip install numpy
```

## Usage
### Create and Train a Neural Network
```python
from pygennet import RandomNeuralNetwork

# Initialize a random neural network
network = RandomNeuralNetwork((3, 5), 5, 2)

# Perform a forward pass
input_data = np.array([1.0, 0.5, -0.5])
output = network.forward(input_data)
print("Network Output:", output)
```

### Mutate and Evolve Networks
```python
from pygennet import genetics

# Generate an initial population
population = generation.createInitialGeneration(10)

# Evaluate and evolve
fitness_scores = [(net, some_fitness_function(net)) for net in population]
top_half = Evolution.select_top_half(fitness_scores)
mutated_population = genetics.mutate_network_parameters(top_half)
```

## Contributing
Feel free to open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License.

