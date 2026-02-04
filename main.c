
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "idx.h"
#include "util.h"
#include "maths.h"

#define TRAIN_FILE "t10k"
#define TRAIN_LABELS "tl10k"

const int single_dim = 28;
const int dim = single_dim * single_dim;

void seed_w1(float w1[10][784])
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 784; ++j)
            w1[i][j] = (randf() - 0.5f) * 0.1f;
}

void seed_w2(float w2[10][10])
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            w2[i][j] = (randf() - 0.5f) * 0.1f;
}

void seed_biases(float b[10])
{
    for (int i = 0; i < 10; ++i)
        b[i] = 0;
}

// matrix multiplication plus activation
// implement with nested loops or vectorize for speed
void forward_propagation(float w1[10][784], float w2[10][10], float b1[10], 
    float b2[10], float X[784], float z1[10], float h[10], float z2[10], float o[10]) 
{
    for (int j = 0; j < 10; ++j)
    {
        z1[j] = 0.0f;
        for (int i = 0; i < 784; ++i)
            z1[j] += w1[j][i] * X[i];
        z1[j] += b1[j];
    }

    for (int i = 0; i < 10; ++i)
        h[i] = relu(z1[i]);

    for (int j = 0; j < 10; ++j)
    {
        z2[j] = 0.0f;
        for (int i = 0; i < 10; ++i)
            z2[j] += w2[j][i] * h[i];
        z2[j] += b2[j];
    }

    softmax(z2, o);
}

void backward_propagation(float w1[10][784], float w2[10][10], float b1[10], 
        float b2[10], float X[784], float z1[10], float h[10], float z2[10], 
        float o[10], float one_hot[10], float lr)
{
    float delta_out[10] = {0};
    float delta_hidden[10] = {0};

    for (int i = 0; i < 10; ++i)
        delta_out[i] = o[i] - one_hot[i];

    // w2 and b2
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            w2[i][j] -= lr * delta_out[i] * h[j]; 
        b2[i] -= lr * delta_out[i];
    }

    // hidden delta
    for (int i = 0; i < 10; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < 10; ++j)
            sum += w2[j][i] * delta_out[j];
        delta_hidden[i] = relu_derivative(z1[i]) * sum;
    }

    // w1 and b1
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 784; ++j)
            w1[i][j] -= lr * delta_hidden[i] * X[j]; 
        b1[i] -= lr * delta_hidden[i];
    }
}

int main(void)
{
    srand(time(0));

    IDX_Data* images = idx_load(TRAIN_FILE);
    log_idx_data(images);   

    IDX_Data* labels = idx_load(TRAIN_LABELS);
    log_idx_data(labels);   

    // log first 10 images and labels
    // perf_test(images, labels);

    float w1[10][784]; // 10x784 (10 neurons per pixel per image)
    float b1[10]; // 10 (per neurons)

    float w2[10][10]; // (10x10 weights per neurons)
    float b2[10]; // 10 (per neurons)

    seed_w1(w1);
    seed_w2(w2);
    seed_biases(b1);
    seed_biases(b2);

    // log_state(w1, w2, b1, b2);

    // work with a single image for now
    uint8_t* this_image = ((uint8_t*)images->data) + 123 * 28 * 28; // take image 123 as example beeing a 6
    uint8_t* this_label = ((uint8_t*)labels->data) + 123; 
                                                      
    // make this this_image pointer a array float [784]
    float float_image[784];
    for (int i = 0; i < 784; ++i)
        float_image[i] = (float)this_image[i] / 255.0f; // 0..1

    float one_hot[10] = {0}; // solution
    one_hot[*this_label] = 1;
    
    float z1[10] = {0};
    float z2[10] = {0};
    float  h[10] = {0};
    float  o[10] = {0};

    forward_propagation(w1, w2, b1, b2, float_image, z1, h, z2, o);
    printf("LOGGING FORWARD PROP\n");
    log_prop(w1, w2, b1, b2, z1, h, z2, o, one_hot);

    const float lr = 1e-2;
    backward_propagation(w1, w2, b1, b2, float_image, z1, h, z2, o, one_hot, lr);
    printf("LOGGING BACKWARD PROP\n");
    log_prop(w1, w2, b1, b2, z1, h, z2, o, one_hot);

    idx_free(images);
    idx_free(labels);

    return 0;
}

