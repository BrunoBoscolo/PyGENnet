#include "evolution.h"
#include <stdlib.h>
#include <stdio.h>

// --- Evolution Functions Implementation ---

// Creates an initial population of neural networks
NeuralNetwork** create_initial_population(int population_size, int num_layers, const int* architecture) {
    NeuralNetwork** population = (NeuralNetwork**)malloc(population_size * sizeof(NeuralNetwork*));
    if (!population) return NULL;

    for (int i = 0; i < population_size; i++) {
        population[i] = create_neural_network(num_layers, architecture);
    }
    return population;
}

// Comparison function for qsort to sort networks by fitness in descending order
int compare_fitness(const void* a, const void* b) {
    const NetworkFitness* nf_a = (const NetworkFitness*)a;
    const NetworkFitness* nf_b = (const NetworkFitness*)b;
    if (nf_a->fitness < nf_b->fitness) return 1;
    if (nf_a->fitness > nf_b->fitness) return -1;
    return 0;
}

// Selects the fittest networks from a population
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
    // Sort the population by fitness
    qsort(population_with_fitness, population_size, sizeof(NetworkFitness), compare_fitness);

    // Select the top half
    *num_fittest = population_size / 2;
    NetworkFitness* fittest = (NetworkFitness*)malloc(*num_fittest * sizeof(NetworkFitness));
    if (!fittest) {
        *num_fittest = 0;
        return NULL;
    }

    for (int i = 0; i < *num_fittest; i++) {
        fittest[i] = population_with_fitness[i];
    }

    return fittest;
}

// Creates a new generation by cloning and mutating the fittest networks
NeuralNetwork** reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, float mutation_rate, float mutation_chance) {
    if (num_fittest == 0) return NULL;

    NeuralNetwork** new_population = (NeuralNetwork**)malloc(new_population_size * sizeof(NeuralNetwork*));
    if (!new_population) return NULL;

    for (int i = 0; i < new_population_size; i++) {
        // Choose a random parent from the fittest networks
        int parent_index = rand() % num_fittest;
        const NeuralNetwork* parent = fittest_networks[parent_index].network;

        // Clone the parent to create a child
        NeuralNetwork* child = clone_network(parent);

        // Mutate the child
        mutate_network(child, mutation_rate, mutation_chance);

        new_population[i] = child;
    }

    return new_population;
}
