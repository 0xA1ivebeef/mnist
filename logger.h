
#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <math.h>

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

void log_neurons_zero(float n[10])
{
    printf("logging neuron state: \n");
    for (int i = 0; i < 10; ++i)
        printf("%d: %f\n", i, n[i]);
    printf("\n\n");
}

void log_prop(float w1[][784], float w2[][10], float b1[], float b2[], float z1[], 
        float h[], float z2[], float o[], float one_hot[])
{
    printf("\n\nlogging state of propagation\n\n");

    // log_state(w1, w2, b1, b2);

    printf("z1\n");
    log_neurons_zero(z1);

    printf("z2\n");
    log_neurons_zero(z2);

    printf("h\n");
    log_neurons_zero(h);

    printf("fp out\n");
    log_neurons_zero(o);

    printf("one hot\n");
    log_neurons_zero(one_hot);

    printf("\n\n");
}

void log_loss(float* one_hot, float* o)
{
    float loss = 0.0f;
    for (int i = 0; i < 10; ++i)
        loss -= one_hot[i]  * logf(o[i] + 1e-8f); // avoid log(0)
    
    printf("log_loss current loss: %f\n", loss);
}

#endif
