
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "idx.h"
#include "util.h"
#include "maths.h"
#include "logger.h"
#include "model.h"
#include "serialize.h"

#define UI_IMPLEMENTATION
#include "ui.h"

#define PROPAGATIONS_IMPLEMETATION
#include "propagations.h"

#define TRAIN_FILE "t60k"
#define TRAIN_LABELS "tl60k"

void check_accuracy(Model* model, Dataset* dataset)
{
    int correct = 0;
        for (int i = 0; i < dataset->image_count; ++i)
        {
            float* X = &dataset->images[i * 784];
            forwardprop(model, X);
            int predicted = argmax_index(model->o, 10);
            if (predicted == dataset->labels[i])
                correct++;
        }
        float accuracy = (float)correct / dataset->image_count;
        printf("Accuracy %.2f%%\n", accuracy * 100);
}

void train_model(Model* model, Dataset* dataset)
{
    seed_model(model);

    BatchModel bm = {0};

    const float lr = 1e-3;
    const int num_epoch = 50;
    for (int epoch = 0; epoch < num_epoch; ++epoch)
    { 
        printf("epoch left: %d\n", num_epoch - epoch);
        
        // with batching
        for (int start = 0; start < dataset->image_count; start += BATCH_SIZE)
        {
            // if image_count not divisible by BATCH_SIZE
            int curr_batch_size = min(BATCH_SIZE, dataset->image_count - start); 

            float* x_batch = &dataset->images[start * 784];
            int* label_batch = &dataset->labels[start];

            batch_forwardprop(model, &bm, x_batch, curr_batch_size);
            batch_backprop(model, &bm, x_batch, label_batch, lr, curr_batch_size);
        }

        // no batching
        /*
        for (int i = 0; i < dataset->image_count; ++i)
        {
            float* X = &dataset->images[i * dataset->image_size];

            forwardprop(model, X);
            backprop(model, X, dataset->labels[i], lr);

            // log_loss(one_hot, o);
        }
        */
    }
}

void get_dataset(Dataset* buff)
{
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

    *buff = (Dataset){images, labels, image_count, image_size};

    idx_free(image_idx);
    idx_free(label_idx);
}

void loop(SDL_Window* window, SDL_Renderer* renderer, Model* model) 
{
    init_sdl(&window, &renderer);

    int running = 1;
    SDL_Event e;
    int mx, my;
    int drag = 0;
    int must_update = 0;

    render_clear(renderer);
    SDL_RenderPresent(renderer);

    while (running)
    {
        SDL_GetMouseState(&mx, &my);
        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_KEYDOWN:
                    if (e.key.keysym.sym == SDLK_ESCAPE)
                        running = 0;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (e.button.button == SDL_BUTTON_LEFT) 
                    {
                        if (button_hit(e.button.x, e.button.y)) 
                            button_clicked();
                    }
                    drag = 1;
                    break;
                case SDL_MOUSEBUTTONUP:
                    drag = 0;
                    break;
            }
        }
        
        if (drag)
        {
            handle_drag(mx, my);
            must_update = 1;
        }

        if (must_update)
        {
            printf("updating\n");
            render_clear(renderer);
            render_state(renderer);

            float image_f[784] = {0};
            float_image_from_uint8(image_f, state);
            // dump_float_image(image_f, dataset->image_size);

            forwardprop(model, image_f);
            printf("number: %d\n", argmax_index(model->o, 10));
            // fflush(stdout);

            SDL_RenderPresent(renderer);
            must_update = 0;
        }
        SDL_Delay(16); // ~60 FPS
    }
}

int main(void)
{
    srand(time(0));

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    
    Dataset dataset;
    get_dataset(&dataset);

    Model model; 
    load_model("out.dat", &model);
    check_accuracy(&model, &dataset);

    loop(window, renderer, &model);

    // save_model(NULL, &model); 

    cleanup_sdl(window, renderer);

    free(dataset.images);
    free(dataset.labels);

    return 0;
}

