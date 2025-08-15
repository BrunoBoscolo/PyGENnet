import numpy as np
from pygennet.models import NeuralNetwork, debugging
from pygennet.evolution import Evolution
from pygennet.data_loader import load_mnist

def preprocess_data(images, labels):
    """
    Preprocesses the MNIST data.
    - Flattens images
    - Normalizes pixel values to [0, 1]
    - One-hot encodes labels
    """
    # Flatten images
    num_images = images.shape[0]
    images_flattened = images.reshape(num_images, -1).astype('float32') / 255.0

    # One-hot encode labels
    labels_one_hot = np.zeros((num_images, 10))
    labels_one_hot[np.arange(num_images), labels] = 1

    return images_flattened, labels_one_hot

def main():
    """
    An example of how to use the pygennet library to evolve a neural network to recognize MNIST digits.
    """
    print("--- Starting MNIST Neural Network Evolution ---")

    # --- 1. Define Parameters ---
    NETWORK_ARCHITECTURE = [784, 128, 10]  # 784 inputs (28x28 image), 128 neurons in hidden layer, 10 outputs (digits 0-9)
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.05
    MUTATION_CHANCE = 0.2
    NUM_GENERATIONS = 20
    FITNESS_BATCH_SIZE = 128

    # --- 2. Load and Preprocess MNIST Data ---
    print("Loading MNIST data...")
    x_train, y_train_raw = load_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    x_test, y_test_raw = load_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')

    x_train_processed, y_train_processed = preprocess_data(x_train, y_train_raw)
    x_test_processed, y_test_processed = preprocess_data(x_test, y_test_raw)
    print("Data loaded and preprocessed.")

    # --- 3. Define a Fitness Function ---
    def calculate_fitness(network: NeuralNetwork) -> float:
        """
        Calculates the fitness of a network based on its accuracy on a random batch of the training data.
        """
        # Select a random batch from the training data
        indices = np.random.choice(len(x_train_processed), FITNESS_BATCH_SIZE, replace=False)
        batch_x = x_train_processed[indices]
        batch_y = y_train_processed[indices]

        # Get network predictions
        predictions = network.forward(batch_x)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(batch_y, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    # --- 4. Create Initial Population ---
    population = Evolution.create_initial_population(
        population_size=POPULATION_SIZE,
        network_architecture=NETWORK_ARCHITECTURE
    )
    print(f"Created initial population of {len(population)} networks.")
    print(f"Network architecture: {NETWORK_ARCHITECTURE}")
    print("-" * 20)

    # --- 5. Run Evolutionary Loop ---
    for gen in range(NUM_GENERATIONS):
        population_with_fitness = [(net, calculate_fitness(net)) for net in population]
        best_network, best_fitness = max(population_with_fitness, key=lambda item: item[1])
        print(f"Generation {gen+1}/{NUM_GENERATIONS} | Best Fitness (Batch Accuracy): {best_fitness:.4f}")

        fittest_networks = Evolution.select_fittest(population_with_fitness)
        population = Evolution.reproduce(
            fittest_networks=fittest_networks,
            new_population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            mutation_chance=MUTATION_CHANCE
        )

    print("-" * 20)
    # --- 6. Final Evaluation ---
    print("Evolution finished. Evaluating the best network on the full test set...")
    final_fitness_scores = [(net, calculate_fitness(net)) for net in population]
    best_overall_network, _ = max(final_fitness_scores, key=lambda item: item[1])

    # Test on the entire test set
    test_predictions = best_overall_network.forward(x_test_processed)
    predicted_labels = np.argmax(test_predictions, axis=1)
    true_labels = np.argmax(y_test_processed, axis=1)
    final_accuracy = np.mean(predicted_labels == true_labels)

    print(f"\nFinal Test Set Accuracy: {final_accuracy:.4f}")

    # Optional: print the parameters of the best network
    # print("\nBest network parameters:")
    # debugging.print_network_parameters(best_overall_network)


if __name__ == "__main__":
    main()
