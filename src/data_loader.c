#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Helper function to swap the byte order of a 32-bit integer
// MNIST data is stored in big-endian format.
static uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
    return (val << 16) | (val >> 16);
}

// Loads a dataset from the specified MNIST IDX files
Dataset* load_dataset(const char* image_path, const char* label_path) {
    // --- Open Files ---
    FILE* image_file = fopen(image_path, "rb");
    if (!image_file) {
        fprintf(stderr, "Error: Could not open image file %s\n", image_path);
        return NULL;
    }
    FILE* label_file = fopen(label_path, "rb");
    if (!label_file) {
        fprintf(stderr, "Error: Could not open label file %s\n", label_path);
        fclose(image_file);
        return NULL;
    }

    // --- Read Image File Header ---
    uint32_t magic_images, num_images, num_rows, num_cols;
    fread(&magic_images, sizeof(uint32_t), 1, image_file);
    fread(&num_images, sizeof(uint32_t), 1, image_file);
    fread(&num_rows, sizeof(uint32_t), 1, image_file);
    fread(&num_cols, sizeof(uint32_t), 1, image_file);

    magic_images = swap_endian(magic_images);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    if (magic_images != 2051) {
        fprintf(stderr, "Error: Invalid magic number in image file %s\n", image_path);
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Read Label File Header ---
    uint32_t magic_labels, num_labels;
    fread(&magic_labels, sizeof(uint32_t), 1, label_file);
    fread(&num_labels, sizeof(uint32_t), 1, label_file);

    magic_labels = swap_endian(magic_labels);
    num_labels = swap_endian(num_labels);

    if (magic_labels != 2049) {
        fprintf(stderr, "Error: Invalid magic number in label file %s\n", label_path);
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    if (num_images != num_labels) {
        fprintf(stderr, "Error: Number of images and labels do not match.\n");
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Allocate Dataset ---
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        fprintf(stderr, "Error: Could not allocate memory for dataset.\n");
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }
    dataset->num_items = num_images;
    int image_size = num_rows * num_cols;
    dataset->images = create_matrix(num_images, image_size);
    dataset->labels = create_matrix(num_images, MNIST_NUM_CLASSES);

    if (!dataset->images || !dataset->labels) {
        fprintf(stderr, "Error: Could not allocate memory for dataset matrices.\n");
        free_dataset(dataset);
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Read Image Data ---
    unsigned char* image_buffer = (unsigned char*)malloc(image_size);
    if (!image_buffer) {
         fprintf(stderr, "Error: Could not allocate memory for image buffer.\n");
         free_dataset(dataset);
         fclose(image_file);
         fclose(label_file);
         return NULL;
    }
    for (int i = 0; i < num_images; i++) {
        fread(image_buffer, sizeof(unsigned char), image_size, image_file);
        for (int j = 0; j < image_size; j++) {
            dataset->images->data[i][j] = (double)image_buffer[j] / 255.0;
        }
    }
    free(image_buffer);

    // --- Read Label Data ---
    unsigned char label_buffer;
    for (int i = 0; i < num_images; i++) {
        fread(&label_buffer, sizeof(unsigned char), 1, label_file);
        for(int j = 0; j < MNIST_NUM_CLASSES; j++) {
            dataset->labels->data[i][j] = 0.0;
        }
        if (label_buffer < MNIST_NUM_CLASSES) {
            dataset->labels->data[i][label_buffer] = 1.0;
        }
    }

    fclose(image_file);
    fclose(label_file);
    printf("Successfully loaded %d items from %s and %s\n", dataset->num_items, image_path, label_path);
    return dataset;
}

// Frees the memory allocated for a dataset
void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    free_matrix(dataset->images);
    free_matrix(dataset->labels);
    free(dataset);
}
