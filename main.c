
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cblas.h>

#include "idx.h"
#include "util.h"
#include "maths.h"
#include "logger.h"
#include "model.h"
#include "serialize.h"

#define UI_IMPLEMENTATION
#include "ui.h"

#define TRAIN_FILE "t60k"
#define TRAIN_LABELS "tl60k"

void forwardprop(Model* model, float* X) 
{
    // s (float) 
    // ge (general) 
    // mv (matrix-vector)
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans, 
        HIDDEN, 784, 1.0f, &model->w1[0][0],    // w1 in1
        784, X, 1, 0.0f,                        // X  in2
        model->z1, 1                            // z1 out
    );

    for (int i = 0; i < HIDDEN; ++i)
        model->h[i] = relu(model->z1[i] + model->b1[i]);

    // z2 = W2 * h
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        10, HIDDEN, 1.0f, &model->w2[0][0], 
        HIDDEN, model->h, 1, 0.0f,
        model->z2, 1
    );

    softmax(model->z2, model->o);
}

void backprop(Model* model, float* X, int label, float lr)
{
    float delta_out[10];
    float delta_hidden[HIDDEN] = {0};

    // Output layer error
    memcpy(delta_out, model->o, sizeof(float) * 10);
    delta_out[label] -= 1;

    // delta_hidden
    cblas_sgemv(
        CblasRowMajor, CblasTrans,
        10, HIDDEN, 1.0f, &model->w2[0][0],
        HIDDEN, delta_out, 1, 0.0f,
        delta_hidden, 1
    );
    for (int i = 0; i < HIDDEN; ++i)
        delta_hidden[i] *= relu_derivative(model->z1[i]);

    // update w2 and b2
    cblas_sger(
        CblasRowMajor,
        10, HIDDEN, -lr, 
        delta_out, 1,        
        model->h, 1,        
        &model->w2[0][0],  
        HIDDEN            
    );
    for (int i = 0; i < 10; ++i) 
        model->b2[i] -= lr * delta_out[i]; 

    // update w1 and b1
    cblas_sger(
        CblasRowMajor, 
        HIDDEN, 784, -lr, 
        delta_hidden, 1,
        X, 1, 
        &model->w1[0][0],
        784
    );
    for (int i = 0; i < HIDDEN; ++i)
        model->b1[i] -= lr * delta_hidden[i];
}

void check_accuracy(Model* model, float* images, const int image_count, const int* labels)
{
    int correct = 0;
        for (int i = 0; i < image_count; ++i)
        {
            float* X = &images[i * 784];
            forwardprop(model, X);
            int predicted = argmax_index(model->o, 10);
            if (predicted == labels[i])
                correct++;
        }
        float accuracy = (float)correct / image_count;
        printf("Accuracy %.2f%%\n", accuracy * 100);
}

int main(void)
{
    // TODO implement batching

    srand(time(0));

    IDX_Data* image_idx = idx_load(TRAIN_FILE);
    log_idx_data(image_idx);

    IDX_Data* label_idx = idx_load(TRAIN_LABELS);
    log_idx_data(label_idx);   

    const int label_count = label_idx->dims[0];
    int* labels = malloc(sizeof(int) * label_count);
    for (int i = 0; i < label_count; ++i)
        labels[i] = *get_label(label_idx, i);

    const int image_count = image_idx->dims[0];
    const int image_size = image_idx->total_bytes / image_count;
    float* images = get_float_images(image_idx);

    float* X = NULL;

    Model model; 
    seed_model(&model);

    const float lr = 1e-3;
    const int num_epoch = 50;
    for (int epoch = 0; epoch < num_epoch; ++epoch)
    { 
        printf("epoch left: %d\n", num_epoch - epoch);
        for (int i = 0; i < image_count; ++i)
        {
            X = &images[i * image_size];

            forwardprop(&model, X);
            backprop(&model, X, labels[i], lr);

            // log_loss(one_hot, o);
        }
    }

    check_accuracy(&model, images, image_count, labels);
    
    save_model(NULL, &model);

    free(images);
    free(labels);

    idx_free(image_idx);
    idx_free(label_idx);

    return 0;
}
