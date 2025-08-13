import numpy as np
from pygennet.models import NeuralNetwork, debugging
from pygennet.evolution import Evolution

def main():
    """
    An example of how to use the pygennet library to evolve a neural network.
    """
    print("--- Starting Neural Network Evolution Example ---")

    # --- 1. Define Parameters ---
    NETWORK_ARCHITECTURE = [3, 5, 2]  # 3 inputs, 1 hidden layer with 5 neurons, 2 outputs
    POPULATION_SIZE = 50
    MUTATION_RATE = 0.05
    MUTATION_CHANCE = 0.1
    NUM_GENERATIONS = 10

    # Define a simple task: learn to output a target vector from a given input
    INPUT_DATA = np.array([[0.5, 0.1, -0.2]])
    TARGET_OUTPUT = np.array([[0.8, 0.3]])

    # --- 2. Define a Fitness Function ---
    def calculate_fitness(network: NeuralNetwork) -> float:
        """
        Calculates the fitness of a network. Higher is better.
        Fitness is the inverse of the Mean Squared Error between the network's output and the target.
        A small epsilon is added to avoid division by zero.
        """
        output = network.forward(INPUT_DATA)
        mse = np.mean((output - TARGET_OUTPUT) ** 2)
        fitness = 1 / (mse + 1e-6)
        return fitness

    # --- 3. Create Initial Population ---
    population = Evolution.create_initial_population(
        population_size=POPULATION_SIZE,
        network_architecture=NETWORK_ARCHITECTURE
    )
    print(f"Created initial population of {len(population)} networks.")
    print(f"Network architecture: {NETWORK_ARCHITECTURE}")
    print("-" * 20)

    # --- 4. Run Evolutionary Loop ---
    for gen in range(NUM_GENERATIONS):
        # Evaluate the fitness of each network in the population
        population_with_fitness = []
        for network in population:
            fitness = calculate_fitness(network)
            population_with_fitness.append((network, fitness))

        # Find the best network of the current generation
        best_network, best_fitness = max(population_with_fitness, key=lambda item: item[1])
        print(f"Generation {gen+1}/{NUM_GENERATIONS} | Best Fitness: {best_fitness:.4f}")

        # Select the fittest individuals
        fittest_networks = Evolution.select_fittest(population_with_fitness)

        # Reproduce to create the next generation
        population = Evolution.reproduce(
            fittest_networks=fittest_networks,
            new_population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            mutation_chance=MUTATION_CHANCE
        )

    print("-" * 20)
    # --- 5. Show Results ---
    # Evaluate the final population's best network
    final_fitness_scores = [(net, calculate_fitness(net)) for net in population]
    best_overall_network, best_overall_fitness = max(final_fitness_scores, key=lambda item: item[1])

    print("Evolution finished.")
    print(f"Best fitness achieved: {best_overall_fitness:.4f}")

    print("\nInspecting the best network's performance:")
    final_output = best_overall_network.forward(INPUT_DATA)
    print(f"Input: {INPUT_DATA[0]}")
    print(f"Target Output: {TARGET_OUTPUT[0]}")
    print(f"Actual Output: [{final_output[0][0]:.4f}, {final_output[0][1]:.4f}]")

    # Optional: print the parameters of the best network
    # print("\nBest network parameters:")
    # debugging.print_network_parameters(best_overall_network)


if __name__ == "__main__":
    main()
