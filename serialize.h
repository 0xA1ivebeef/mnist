
#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <stdio.h>
#include <error.h>
#include <stdint.h>

#include "model.h"

/*
    int load_model(const char* filename, Model* model);
    int save_model(char* filename, Model* model);
    void save_pgm(char* filename, uint8_t* pixels);
*/

int load_model(const char *filename, Model *model) 
{
    if (!filename || !model)
    {
        fprintf(stderr, "load_model input error\n");
        return -1;
    }

    FILE *f = fopen(filename, "rb"); // read binary
    if (!f) 
    {
        perror("load_model fopen");
        return -1;
    }

    if (fread(model, sizeof(Model), 1, f) != 1) 
    {
        perror("load_model fread");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

int save_model(char* filename, Model* model)
{
    if (!model)
    {
        printf("save_model invalid input\n");
        return -1;
    }

    if (!filename)
        filename = "out.dat";

    FILE* f = fopen(filename, "a");
    if (!f)
    {
        perror("save_model fopen\n");
        return -1;
    }
    
    if (fwrite(model, sizeof(Model), 1, f) != 1)
    {
        fprintf(stderr, "fwrite\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
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

