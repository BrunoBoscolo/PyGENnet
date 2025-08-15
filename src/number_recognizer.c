#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neural_network.h"
#include "evolution.h"
#include "data_loader.h"

// --- Fitness Function for Classification ---
// Calculates fitness based on classification accuracy on a batch of data.
double calculate_classification_fitness(NeuralNetwork* network, const Dataset* dataset) {
    int correct_predictions = 0;
    for (int i = 0; i < dataset->num_items; i++) {
        // Create a matrix for the current image
        Matrix image_matrix = {1, MNIST_IMAGE_SIZE, &dataset->images->data[i]};

        Matrix* output = forward_pass(network, &image_matrix);
        if (!output) continue;

        // Find the index of the max value in the output (predicted class)
        int predicted_class = 0;
        double max_val = output->data[0][0];
        for (int j = 1; j < MNIST_NUM_CLASSES; j++) {
            if (output->data[0][j] > max_val) {
                max_val = output->data[0][j];
                predicted_class = j;
            }
        }

        // Find the index of the max value in the label (true class)
        int true_class = 0;
        for (int j = 1; j < MNIST_NUM_CLASSES; j++) {
            if (dataset->labels->data[i][j] > dataset->labels->data[i][true_class]) {
                true_class = j;
            }
        }

        if (predicted_class == true_class) {
            correct_predictions++;
        }

        free_matrix(output);
    }

    return (double)correct_predictions / dataset->num_items;
}

int main() {
    printf("--- Starting Number Recognition Training ---\n");

    // --- 1. Define Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    const int NUM_LAYERS = sizeof(ARCHITECTURE) / sizeof(int);
    const int POPULATION_SIZE = 50;
    const float MUTATION_RATE = 0.1f;
    const float MUTATION_CHANCE = 0.2f;
    const int NUM_GENERATIONS = 20; // Reduced for a quicker run

    // --- 2. Load Training Dataset ---
    printf("Loading MNIST training data...\n");
    Dataset* train_dataset = load_dataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    if (!train_dataset) {
        fprintf(stderr, "Failed to load training dataset.\n");
        return 1;
    }

    // --- 3. Create Initial Population ---
    srand(time(NULL));
    NeuralNetwork** population = create_initial_population(POPULATION_SIZE, NUM_LAYERS, ARCHITECTURE);
    printf("Created initial population of %d networks.\n", POPULATION_SIZE);
    printf("----------------------------------------\n");

    // --- 4. Run Evolutionary Loop ---
    for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
        NetworkFitness population_with_fitness[POPULATION_SIZE];
        double best_fitness_in_gen = 0.0;

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population_with_fitness[i].network = population[i];
            population_with_fitness[i].fitness = calculate_classification_fitness(population[i], train_dataset);
            if (population_with_fitness[i].fitness > best_fitness_in_gen) {
                best_fitness_in_gen = population_with_fitness[i].fitness;
            }
        }
        printf("Generation %d/%d | Best Training Accuracy: %.4f\n", gen + 1, NUM_GENERATIONS, best_fitness_in_gen);

        int num_fittest;
        NetworkFitness* fittest_networks_info = select_fittest(population_with_fitness, POPULATION_SIZE, &num_fittest);
        NeuralNetwork** new_population = reproduce(fittest_networks_info, num_fittest, POPULATION_SIZE, MUTATION_RATE, MUTATION_CHANCE);

        for (int i = 0; i < POPULATION_SIZE; i++) {
            free_neural_network(population[i]);
        }
        free(population);
        free(fittest_networks_info);
        population = new_population;
    }
    printf("----------------------------------------\n");
    printf("Evolution finished.\n\n");

    // --- 5. Evaluate on Test Data ---
    printf("Loading MNIST test data...\n");
    Dataset* test_dataset = load_dataset("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    if (!test_dataset) {
        fprintf(stderr, "Failed to load test dataset.\n");
        free_dataset(train_dataset);
        for (int i = 0; i < POPULATION_SIZE; i++) free_neural_network(population[i]);
        free(population);
        return 1;
    }

    // Find the best network from the final population
    NeuralNetwork* best_network = population[0];
    double best_fitness = calculate_classification_fitness(best_network, train_dataset);
    for (int i = 1; i < POPULATION_SIZE; i++) {
        double current_fitness = calculate_classification_fitness(population[i], train_dataset);
        if (current_fitness > best_fitness) {
            best_fitness = current_fitness;
            best_network = population[i];
        }
    }

    // Test the best network
    double test_accuracy = calculate_classification_fitness(best_network, test_dataset);
    printf("----------------------------------------\n");
    printf("Final Test Accuracy of the best network: %.4f\n", test_accuracy);
    printf("----------------------------------------\n");

    // --- 6. Cleanup ---
    printf("Cleaning up resources...\n");
    free_dataset(train_dataset);
    free_dataset(test_dataset);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        free_neural_network(population[i]);
    }
    free(population);
    printf("Done.\n");

    return 0;
}
