
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "idx.h"
#include "maths.h"

#define HIDDEN 128

const char* ascii_char = ".:-=+*#%@";

const float float_min = -2147483648.0f;
const float float_max = 2147483648.0f;

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b

/*
uint8_t* get_image(IDX_Data* images, const int index);
uint8_t* get_label(IDX_Data* labels, const int index);
void dump_float_image(const float* data, const int px_count);
void dump_pgm(uint8_t* data, int px_count);
void perf_test(IDX_Data* images, IDX_Data* labels);
void float_image_from_uint8(float* float_image, uint8_t* this_image);
void get_one_hot(float* one_hot, const int l);
float** get_float_images(IDX_Data* images);
int argmax_index(const float* o, const int n);
void save_pgm(char* filename, uint8_t* pixels);
*/

void seed_w1(float w1[HIDDEN][784])
{
    for (int i = 0; i < HIDDEN; ++i)
        for (int j = 0; j < 784; ++j)
            w1[i][j] = (randf() - 0.5f) * 0.1f;
}

void seed_w2(float w2[10][HIDDEN])
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < HIDDEN; ++j)
            w2[i][j] = (randf() - 0.5f) * 0.1f;
}

void seed_biases(float* b, int size)
{
    for (int i = 0; i < size; ++i)
        b[i] = 0;
}

uint8_t* get_image(IDX_Data* images, const int index)
{
    if (index >= images->dims[0])
        return NULL;

    return ((uint8_t*)images->data) + index * 784;   
}

uint8_t* get_label(IDX_Data* labels, const int index)
{
    if (index >= labels->dims[0])
        return NULL;

    return ((uint8_t*)labels->data) + index;
}

void dump_float_image(const float* data, const int px_count)
{
    if (!data || px_count < 1)
    {
        fprintf(stderr, "ERROR, dump_float_image invalid input \n");
        return;
    }
    
    const int ascii_char_l = strlen(ascii_char);
    for (int i = 0; i < px_count; ++i)
    {
        float px = data[i];

        int idx = (px * ascii_char_l);
        printf("%c", ascii_char[idx]);
                
        if ((i + 1) % (int)sqrt(px_count) == 0)
            printf("\n");
    }
    
}

void dump_pgm(const uint8_t* data, const int px_count)
{
    if (!data || px_count < 1)
    {
        fprintf(stderr, "ERROR, dump_pgm invalid input\n");
        return;
    }
    
    const int ascii_char_l = strlen(ascii_char);
    for (int i = 0; i < px_count; ++i)
    {
        uint8_t px = data[i];

        int idx = (px * ascii_char_l) / 255;
        printf("%c", ascii_char[idx]);
                
        if ((i + 1) % (int)sqrt(px_count) == 0)
            printf("\n");
    }
}

void perf_test(IDX_Data* images, IDX_Data* labels)
{
    uint8_t* label123 = ((uint8_t*)labels->data) + 123;
    printf("label 123: %d\n", *label123);

    for (int i = 10; i < 20; ++i)
    {
        uint8_t* img = get_image(images, i);
        dump_pgm(img, images->dims[0]);

        uint8_t l = *get_label(labels, i);
        printf("index %d, label: %d\n", i, l);
    }
}

int argmax_index(const float* o, const int n)
{
    if (!o || n < 1)
    {
        fprintf(stderr, "argmax_index input error\n");
        return -1; 
    }

    float m = float_min;
    int idx;
    for (int i = 0; i < n; ++i)
    {
        if (o[i] > m)
        {
            m = o[i];
            idx = i;
        }
    }
    return idx;
}

void float_image_from_uint8(float* float_image, uint8_t* this_image)
{
    for (int i = 0; i < 784; ++i)
        float_image[i] = (float)this_image[i] / 255.0f; // 0..1
}

void get_one_hot(float* one_hot, const int l)
{
    for (int i = 0; i < 10; ++i)
        one_hot[i] = 0;

    one_hot[l] = 1;
}

float* get_float_images(IDX_Data* images)
{
    if (!images)
    {
        fprintf(stderr, "get_float_images input error\n");
        return NULL;
    }

    float* float_images = malloc(sizeof(float) * images->total_bytes); 
    if (!float_images)
    {
        fprintf(stderr, "get_float_images malloc failed\n");
        return NULL;
    }

    const int image_count = images->dims[0];
    float X[784] = {0};
    for (int i = 0; i < image_count; ++i)
    {
        float_image_from_uint8(X, get_image(images, i));
        memcpy(float_images + i*784, X, sizeof(X));
    }
    return float_images;
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
