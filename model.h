
#ifndef MODEL_H
#define MODEL_H

#define HIDDEN 128
#define BATCH_SIZE 32

typedef struct Model
{
    float   w1[HIDDEN][784];
    float   b1[HIDDEN]; 
    float   w2[10][HIDDEN]; 
    float   b2[10]; 
    float   z1[HIDDEN]; 
    float   h[HIDDEN]; 
    float   z2[10]; 
    float   o[10];
} Model;

typedef struct BatchModel 
{
    float z1_batch[BATCH_SIZE * HIDDEN];
    float h_batch[BATCH_SIZE * HIDDEN];
    float z2_batch[BATCH_SIZE * 10];
    float o_batch[BATCH_SIZE * 10];
} BatchModel;

#endif
