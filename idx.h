
#ifndef IDX_H
#define IDX_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

typedef struct Dataset
{
    float* images;
    int* labels;
    int image_count;
    int image_size;
} Dataset;

typedef struct 
{
    uint8_t data_type;     // e.g. 0x08 = uint8_t
    uint8_t num_dims;      // usually 1 for labels, 3 for images
    uint32_t dims[8];      // dimension sizes
    void* data;            // raw pointer to payload (flat array)
    size_t total_bytes;    // size of payload in bytes
} IDX_Data;

void log_idx_data(IDX_Data* d)
{
    if (!d)
    {
        fprintf(stderr, "ERROR: log_idx_data Invalid IDX Data passed\n"); 
        return;
    }

    printf("--------------------------\n");
    printf("Logging IDX Data:\n");
    printf("\n");

    printf("Data type: 0x%02X\n", d->data_type);
    printf("Number of Dimensions: %d\n", d->num_dims);

    printf("(max printing for dimensions 8)\n");
    for (size_t i = 0; i < d->num_dims && i < 8; ++i)
        printf("Size of Dimension %ld: %d\n", i, d->dims[i]);

    printf("Size of payload: %ld bytes\n", d->total_bytes);

    printf("--------------------------\n\n");
}

static uint32_t read_be_uint32(uint8_t b[4]) 
{
    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |
            (uint32_t)b[3];
}

IDX_Data* idx_load(const char* filename) 
{
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    uint8_t header[4];
    if (fread(header, 1, 4, f) != 4) 
    {
        fclose(f);
        return NULL;
    }

    IDX_Data* idx  = calloc(1, sizeof(IDX_Data));
    idx->data_type = header[2];   // byte 2
    idx->num_dims  = header[3];   // byte 3

    // Read dimension sizes (4 bytes per dimension)
    for (int i = 0; i < idx->num_dims; i++) 
    {
        uint8_t dimbuf[4];
        fread(dimbuf, 1, 4, f);
        idx->dims[i] = read_be_uint32(dimbuf);
    }

    // Compute total payload size
    size_t element_size = 0;
    switch(idx->data_type) 
    {
        case 0x08: element_size = 1; break; // uint8_t
        case 0x09: element_size = 1; break; // int8_t
        case 0x0B: element_size = 2; break; // int16
        case 0x0C: element_size = 4; break; // int32
        case 0x0D: element_size = 4; break; // float
        case 0x0E: element_size = 8; break; // double
        default:
            fclose(f);
            free(idx);
            return NULL;
    }

    uint64_t total_elements = 1;
    for (int i = 0; i < idx->num_dims; i++) 
        total_elements *= idx->dims[i];

    idx->total_bytes = total_elements * element_size;
    idx->data = malloc(idx->total_bytes);

    fread(idx->data, 1, idx->total_bytes, f);
    fclose(f);

    return idx;
}

void idx_free(IDX_Data* idx) 
{
    if (!idx) return;
    free(idx->data);
    free(idx);
}

#endif

