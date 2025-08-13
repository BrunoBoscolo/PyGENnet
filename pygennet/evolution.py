import numpy as np
from .models import NeuralNetwork

class Evolution:
    """
    Provides methods to run an evolutionary algorithm on NeuralNetworks.
    """

    @staticmethod
    def create_initial_population(population_size: int, network_architecture: list[int]) -> list[NeuralNetwork]:
        """
        Creates an initial population of random neural networks.
        """
        population = []
        for _ in range(population_size):
            network = NeuralNetwork(architecture=network_architecture)
            population.append(network)
        return population

    @staticmethod
    def select_fittest(population_with_fitness: list[tuple[NeuralNetwork, float]]) -> list[NeuralNetwork]:
        """
        Selects the top-performing half of the population based on fitness scores.
        'population_with_fitness' should be a list of (network, fitness_score) tuples.
        """
        # Sort networks by fitness in descending order (higher is better)
        sorted_population = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

        # Select the top half
        num_to_select = len(sorted_population) // 2
        fittest_networks = [item[0] for item in sorted_population[:num_to_select]]

        return fittest_networks

    @staticmethod
    def reproduce(fittest_networks: list[NeuralNetwork], new_population_size: int, mutation_rate: float, mutation_chance: float) -> list[NeuralNetwork]:
        """
        Creates a new generation by cloning and mutating the fittest networks.
        """
        new_population = []

        # Ensure we have some networks to reproduce from
        if not fittest_networks:
            # If the fittest list is empty, we can't reproduce.
            # This might happen if selection returns nothing.
            # We could either raise an error or return an empty list.
            return []

        while len(new_population) < new_population_size:
            # Choose a parent from the fittest networks at random
            parent = np.random.choice(fittest_networks)

            # Create a child by cloning the parent's parameters
            child = NeuralNetwork(architecture=parent.architecture)
            child.set_parameters(parent.get_parameters())

            # Mutate the child
            child.mutate(mutation_rate=mutation_rate, mutation_chance=mutation_chance)

            new_population.append(child)

        return new_population
