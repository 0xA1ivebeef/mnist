
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "idx.h"

extern const int single_dim;
extern const int dim;

const char* ascii_char = ".:-=+*#%@";

/*
uint8_t* get_image(IDX_Data* images, const int image_count, const int index);
uint8_t* get_label(IDX_Data* labels, const int image_count, const int index);
void dump_pgm(uint8_t* data, int px_count);
void perf_test(IDX_Data* images, IDX_Data* labels);
void log_state(float w1[10][784], float w2[10][10], float b1[10], float b2[10]);
void log_neurons(float n[10]);
void save_pgm(char* filename, uint8_t* pixels);
*/

uint8_t* get_image(IDX_Data* images, const int image_count, const int index)
{
    if (index >= image_count)
        return NULL;

    return ((uint8_t*)images->data) + index * dim;   
}

uint8_t* get_label(IDX_Data* labels, const int image_count, const int index)
{
    if (index >= image_count)
        return NULL;

    return ((uint8_t*)labels->data) + index;
}

void dump_pgm(uint8_t* data, int px_count)
{
    if (!data)
    {
        fprintf(stderr, "ERROR, dump_pgm given empty data array\n");
        return;
    }
    
    const int ascii_char_l = strlen(ascii_char);
    for (int i = 0; i < px_count; ++i)
    {
        uint8_t px = data[i];

        int idx = (px * ascii_char_l) / 255;
        printf("%c", ascii_char[idx]);
                
        if ((i + 1) % single_dim == 0)
            printf("\n");
    }
}

void perf_test(IDX_Data* images, IDX_Data* labels)
{
    uint8_t* label123 = ((uint8_t*)labels->data) + 123;
    printf("label 123: %d\n", *label123);

    for (int i = 10; i < 20; ++i)
    {
        uint8_t* img = get_image(images, images->total_bytes / dim, i);
        dump_pgm(img, dim);

        uint8_t l = *get_label(labels, images->total_bytes / dim, i);
        printf("index %d, label: %d\n", i, l);
    }
}

void log_state(float w1[10][784], float w2[10][10], float b1[10], float b2[10])
{
    // w1
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 784; ++j)
            printf("%f, ", w1[i][j]);
        printf("\n");
    }

    // w2
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            printf("%f, ", w2[i][j]);
        printf("\n");
    }
    
    // biases b1 and b2 
    printf("b1:\n");
    for (int i = 0; i < 10; ++i)
        printf("%f, ", b1[i]);
    printf("\n\n b2:");

    for (int i = 0; i < 10; ++i)
        printf("%f, ", b2[i]);

    printf("\n\n");
}

void log_neurons(float n[10])
{
    printf("logging neuron state: \n");
    for (int i = 0; i < 10; ++i)
        printf("%d: %f\n", i+1, n[i]);
    printf("\n\n");
}

void log_prop(float w1[][784], float w2[][10], float b1[], float b2[], float z1[], 
        float h[], float z2[], float o[], float one_hot[])
{
    printf("\n\nlogging state of propagation\n\n");

    // log_state(w1, w2, b1, b2);

    printf("z1\n");
    log_neurons(z1);

    printf("z2\n");
    log_neurons(z2);

    printf("h\n");
    log_neurons(h);

    printf("fp out\n");
    log_neurons(o);

    printf("one hot\n");
    log_neurons(one_hot);

    printf("\n\n");
}

void save_pgm(char* filename, uint8_t* pixels) 
{
    if (!filename)
        filename = "out.pgm";

    if (!pixels)
    {
        fprintf(stderr, "ERROR: save_pgm given empty data_array\n");
        return;
    }

    FILE* f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "save_to_pgm failed to open file\n");
        return;
    }

    fprintf(f, "P5\n28 28\n255\n");
    fwrite(pixels, 1, 28 * 28, f);
    fclose(f);
}

#endif
