#include "data_loader.h"
#include <stdlib.h>
#include <time.h>

// Creates a dummy dataset with random values
Dataset* create_dummy_dataset(int num_items) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) return NULL;

    dataset->num_items = num_items;
    dataset->images = create_matrix(num_items, MNIST_IMAGE_SIZE);
    dataset->labels = create_matrix(num_items, MNIST_NUM_CLASSES);

    if (!dataset->images || !dataset->labels) {
        free_matrix(dataset->images);
        free_matrix(dataset->labels);
        free(dataset);
        return NULL;
    }

    // Seed random number generator if not already seeded
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

    // Fill images with random pixel values (0.0 to 1.0)
    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            dataset->images->data[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Fill labels with random one-hot encoded vectors
    for (int i = 0; i < num_items; i++) {
        int random_class = rand() % MNIST_NUM_CLASSES;
        dataset->labels->data[i][random_class] = 1.0;
    }

    return dataset;
}

// Frees the memory allocated for a dataset
void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    free_matrix(dataset->images);
    free_matrix(dataset->labels);
    free(dataset);
}
