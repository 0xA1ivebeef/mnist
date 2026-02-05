
#ifndef MATHS_H
#define MATHS_H

#include <math.h>
#include <stdlib.h>

static inline float relu(float x)
{
    if (x > 0.0f)
        return x;
    return 0.0f;
}

static inline float relu_derivative(float x)
{
    if (x > 0.0f)
        return 1.0f;
    return 0.0f;
}

void softmax(const float z[10], float o[10])
{
    float max_z = z[0];
    for (int i = 1; i < 10; ++i)
        if (z[i] > max_z) max_z = z[i];

    float sum = 0.0f;
    for (int i = 0; i < 10; ++i)
    {
        o[i] = expf(z[i] - max_z);
        sum += o[i];
    }

    for (int i = 0; i < 10; ++i)
        o[i] /= sum;
}

static inline float dot_product(const float* a, const float* b, int n) 
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) 
        sum += a[i] * b[i];
    return sum;
}

static inline float randf()
{
    return (float)rand() / RAND_MAX;
}

#endif
