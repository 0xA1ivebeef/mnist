
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "idx.h"
#include "maths.h"
#include "model.h"

const float float_min = -2147483648.0f;
const float float_max = 2147483648.0f;

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b

/*
    seed_model(Model* model);
 
    uint8_t* get_image(IDX_Data* images, const int index);
    uint8_t* get_label(IDX_Data* labels, const int index);

    void float_image_from_uint8(float* float_image, uint8_t* this_image);
    float* get_float_images(IDX_Data* images);

    int argmax_index(const float* o, const int n);
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

void seed_model(Model* model)
{
    seed_w1(model->w1);
    seed_w2(model->w2);
    seed_biases(model-> b1, HIDDEN);
    seed_biases(model-> b2, 10);
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

#endif
