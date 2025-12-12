
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct 
{
    uint8_t data_type;     // e.g. 0x08 = uint8_t
    uint8_t num_dims;      // usually 1 for labels, 3 for images
    uint32_t dims[8];      // dimension sizes
    void* data;            // raw pointer to payload (flat array)
    size_t total_bytes;    // size of payload in bytes
} IDX_Data;

float relu(float x)
{
    if (x > 0)
        return x;
    return 0;
}

float rand_f()
{
    return ((float)rand() / (float)RAND_MAX) - 0.5f;
}

void seed_weights(float weights[][784])
{
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 784; ++j)
            weights[i][j] = rand_f();
}

void seed_biases(float b[])
{
    for (int i = 0; i < 10; ++i)
        b[i] = 0; // TODO for now (might adjust)
}

void save_pgm(const char* filename, uint8_t* pixels) 
{
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n28 28\n255\n");
    fwrite(pixels, 1, 28 * 28, f);
    fclose(f);
}

void log_idx_data(IDX_Data* d)
{
    if (!d)
    {
        perror("Invalid IDX Data passed");
        exit(1);
    }

    printf("--------------------------\n");
    printf("Logging IDX Data:\n");
    printf("\n");

    printf("Data type: 0x%02X\n", d->data_type);
    printf("Number of Dimensions: %d\n", d->num_dims);

    printf("(max printing for dimensions 8)\n");
    for (size_t i = 0; i < d->num_dims && i < 8; ++i)
        printf("Size of Dimension %d: %d\n", i, d->dims[i]);

    printf("Size of payload: %d bytes\n", d->total_bytes);

    printf("--------------------------\n\n");
}

static uint32_t read_be_uint32(uint8_t b[4]) 
{
    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |
            (uint32_t)b[3];
}

