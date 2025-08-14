#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neural_network.h"
#include "evolution.h"

// --- Fitness Function ---
double calculate_fitness(NeuralNetwork* network, const Matrix* input_data, const Matrix* target_output) {
    Matrix* output = forward_pass(network, input_data);
    if (!output) {
        fprintf(stderr, "Forward pass failed.\n");
        return 0.0;
    }

    double mse = 0.0;
    for (int i = 0; i < target_output->rows; i++) {
        for (int j = 0; j < target_output->cols; j++) {
            double error = output->data[i][j] - target_output->data[i][j];
            mse += error * error;
        }
    }
    mse /= (target_output->rows * target_output->cols);

    free_matrix(output);

    return 1.0 / (mse + 1e-6);
}

int main() {
    printf("--- Starting Neural Network Evolution Example (C Version) ---\n");

    // --- 1. Define Parameters ---
    const int ARCHITECTURE[] = {3, 5, 2};
    const int NUM_LAYERS = sizeof(ARCHITECTURE) / sizeof(int);
    const int POPULATION_SIZE = 50;
    const float MUTATION_RATE = 0.05f;
    const float MUTATION_CHANCE = 0.1f;
    const int NUM_GENERATIONS = 10;

    // --- Task Definition ---
    Matrix* input_data = create_matrix(1, 3);
    input_data->data[0][0] = 0.5;
    input_data->data[0][1] = 0.1;
    input_data->data[0][2] = -0.2;

    Matrix* target_output = create_matrix(1, 2);
    target_output->data[0][0] = 0.8;
    target_output->data[0][1] = 0.3;

    // --- 3. Create Initial Population ---
    srand(time(NULL)); // Seed for evolution randomness
    NeuralNetwork** population = create_initial_population(POPULATION_SIZE, NUM_LAYERS, ARCHITECTURE);
    printf("Created initial population of %d networks.\n", POPULATION_SIZE);
    printf("Network architecture: [");
    for(int i=0; i<NUM_LAYERS; i++) printf("%d%s", ARCHITECTURE[i], i == NUM_LAYERS - 1 ? "" : ", ");
    printf("]\n");
    printf("--------------------\n");

    // --- 4. Run Evolutionary Loop ---
    for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
        NetworkFitness population_with_fitness[POPULATION_SIZE];
        double best_fitness_in_gen = 0.0;

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population_with_fitness[i].network = population[i];
            population_with_fitness[i].fitness = calculate_fitness(population[i], input_data, target_output);
            if (population_with_fitness[i].fitness > best_fitness_in_gen) {
                best_fitness_in_gen = population_with_fitness[i].fitness;
            }
        }
        printf("Generation %d/%d | Best Fitness: %.4f\n", gen + 1, NUM_GENERATIONS, best_fitness_in_gen);

        int num_fittest;
        NetworkFitness* fittest_networks_info = select_fittest(population_with_fitness, POPULATION_SIZE, &num_fittest);

        NeuralNetwork** new_population = reproduce(fittest_networks_info, num_fittest, POPULATION_SIZE, MUTATION_RATE, MUTATION_CHANCE);

        // Free the old population's networks
        for (int i = 0; i < POPULATION_SIZE; i++) {
            free_neural_network(population[i]);
        }
        free(population);

        // The fittest_networks_info contains pointers to networks that are now freed.
        // We don't need to free the networks inside fittest_networks_info again.
        free(fittest_networks_info);

        population = new_population;
    }

    printf("--------------------\n");
    // --- 5. Show Results ---
    NetworkFitness final_population_fitness[POPULATION_SIZE];
    double best_overall_fitness = 0.0;
    NeuralNetwork* best_overall_network = NULL;

    for (int i = 0; i < POPULATION_SIZE; i++) {
        final_population_fitness[i].network = population[i];
        final_population_fitness[i].fitness = calculate_fitness(population[i], input_data, target_output);
        if (final_population_fitness[i].fitness > best_overall_fitness) {
            best_overall_fitness = final_population_fitness[i].fitness;
            best_overall_network = population[i];
        }
    }

    printf("Evolution finished.\n");
    printf("Best fitness achieved: %.4f\n", best_overall_fitness);

    printf("\nInspecting the best network's performance:\n");
    Matrix* final_output = forward_pass(best_overall_network, input_data);
    printf("Input: [%.2f, %.2f, %.2f]\n", input_data->data[0][0], input_data->data[0][1], input_data->data[0][2]);
    printf("Target Output: [%.2f, %.2f]\n", target_output->data[0][0], target_output->data[0][1]);
    printf("Actual Output: [%.4f, %.4f]\n", final_output->data[0][0], final_output->data[0][1]);

    // --- 6. Cleanup ---
    free_matrix(input_data);
    free_matrix(target_output);
    free_matrix(final_output);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        free_neural_network(population[i]);
    }
    free(population);

    return 0;
}
