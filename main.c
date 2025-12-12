
#include "main.h"

#define TRAIN_FILE "t10k"
#define TRAIN_LABELS "tl10k"

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

    IDX_Data* idx = calloc(1, sizeof(IDX_Data));
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

float dot_product(const float* a, const float* b, int n) 
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) 
        sum += a[i] * b[i];
    return sum;
}

// give all parameters and an input vector X[784]
void forward_propagation(float w1[][784], float b1[], float w2[][10], float b2, float X[]) 
{
    
}


int main(void)
{
    srand(time(0));

    IDX_Data* images = idx_load(TRAIN_FILE);
    log_idx_data(images);   

    IDX_Data* labels = idx_load(TRAIN_LABELS);
    log_idx_data(labels);   

    /* TEST:
    uint8_t* img123 = ((uint8_t*)images->data) + 123 * 28 * 28;
    save_pgm("image123.pgm", img123);

    uint8_t* label123 = ((uint8_t*)labels->data) + 123;
    printf("label 123: %d\n", *label123);
    */ 

    int num_neurons = 10;
    int num_pixels = 28 * 28; // 784
    
    // each neuron needs a vector of weights per pixel which should be random
    float w1[num_neurons][num_pixels];
    float b1[num_neurons];

    float w2[num_neurons][num_neurons];
    float b2[num_neurons];

    seed_weights(w1);
    seed_biases(b1);

    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            w2[i][j] = rand_f();

    seed_biases(b2);
    
    idx_free(images);
    idx_free(labels);

    return 0;
}

