
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

void batch_forwardprop(Model* model, BatchModel* bm, float* x_batch, const int curr_batch_size) 
{
    // Input -> hidden
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        curr_batch_size,   // <-- use curr_batch_size
        HIDDEN, 784, 1.0f,
        x_batch, 784,
        &model->w1[0][0], 784,
        0.0f,
        bm->z1_batch, HIDDEN
    );

    // Add bias + ReLU
    for (int i = 0; i < curr_batch_size; ++i)
        for (int j = 0; j < HIDDEN; ++j)
            bm->h_batch[i * HIDDEN + j] = relu(bm->z1_batch[i * HIDDEN + j] + model->b1[j]);

    // Hidden -> output
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        curr_batch_size,  // <-- use curr_batch_size
        10, HIDDEN, 1.0f,
        bm->h_batch, HIDDEN,
        &model->w2[0][0], HIDDEN,
        0.0f,
        bm->z2_batch, 10
    );

    // Softmax row-wise
    for (int i = 0; i < curr_batch_size; ++i)
        softmax(&bm->z2_batch[i * 10], &bm->o_batch[i * 10]);
}

void batch_backprop(Model* model, BatchModel* bm, float* x_batch, int* label_batch, const float lr, const int curr_batch_size)
{
    float delta_out_batch[BATCH_SIZE * 10];
    float delta_hidden_batch[BATCH_SIZE * HIDDEN];

    // Output layer error
    memcpy(delta_out_batch, bm->o_batch, sizeof(float) * curr_batch_size * 10);
    for (int i = 0; i < curr_batch_size; ++i)
        delta_out_batch[i * 10 + label_batch[i]] -= 1.0f;

    // delta_hidden
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        curr_batch_size,
        HIDDEN, 10, 1.0f, delta_out_batch, 
        10, &model->w2[0][0], 
        HIDDEN, 0.0f, delta_hidden_batch, 
        HIDDEN
    );   
    for (int i = 0; i < curr_batch_size; ++i)
        for (int j = 0; j < HIDDEN; ++j)
            delta_hidden_batch[i*HIDDEN + j] *= relu_derivative(bm->z1_batch[i*HIDDEN + j]);

    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        10, HIDDEN, curr_batch_size,
        -lr,
        delta_out_batch, 10,
        bm->h_batch, HIDDEN, 1.0f,
        &model->w2[0][0], HIDDEN
    );

    for (int j = 0; j < 10; ++j) 
    {
        float sum = 0.0f;
        for (int i = 0; i < curr_batch_size; ++i)
            sum += delta_out_batch[i*10 + j];
        model->b2[j] -= lr * sum;
    }

    // update w1 and b1
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        HIDDEN, 784, curr_batch_size,
        -lr,
        delta_hidden_batch, HIDDEN,
        x_batch, 784, 1.0f,
        &model->w1[0][0], 784
    );   

    for (int j = 0; j < HIDDEN; ++j) 
    {
        float sum = 0.0f;
        for (int i = 0; i < curr_batch_size; ++i)
            sum += delta_hidden_batch[i*HIDDEN + j];
        model->b1[j] -= lr * sum;
    }
}

void forwardprop(Model* model, float* X) 
{
    // s (float) 
    // ge (general) 
    // mv (matrix-vector)
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans, 
        HIDDEN, 784, 1.0f, &model->w1[0][0],    // w1 in1
        784, X, 1, 0.0f,                        // X  in2 single 784 vector
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

    /* training model
    seed_model(&model);

    BatchModel bm = {0};

    const float lr = 1e-3;
    const int num_epoch = 50;
    for (int epoch = 0; epoch < num_epoch; ++epoch)
    { 
        printf("epoch left: %d\n", num_epoch - epoch);
        
        //  yes batching :)
        for (int start = 0; start < image_count; start += BATCH_SIZE)
        {
            // if image_count not divisible by BATCH_SIZE
            int curr_batch_size = min(BATCH_SIZE, image_count - start); 

            float* x_batch = &images[start * 784];
            int* label_batch = &labels[start];

            batch_forwardprop(&model, &bm, x_batch, curr_batch_size);
            batch_backprop(&model, &bm, x_batch, label_batch, lr, curr_batch_size);
        }
        */
        /*

         no batching
        for (int i = 0; i < image_count; ++i)
        {
            X = &images[i * image_size];

            forwardprop(&model, X);
            backprop(&model, X, labels[i], lr);

            // log_loss(one_hot, o);
        }
        
        */
    // }

    load_model("out.dat", &model);
    check_accuracy(&model, images, image_count, labels);
 
    

    // save_model(NULL, &model);

    free(images);
    free(labels);

    idx_free(image_idx);
    idx_free(label_idx);

    return 0;
}

