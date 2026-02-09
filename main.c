
#define UI_IMPLEMENTATION
#include "ui.h"

#include <stdio.h>

typedef struct Button
{
    int x, y;
    int w, h;
} Button;

#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b

Button b = {0, 0, 200, 50};

#define BUFFER_WIDTH 28
#define BUFFER_HEIGHT 28
#define BUFFER_SIZE BUFFER_WIDTH*BUFFER_HEIGHT
uint8_t state[BUFFER_SIZE] = {0};

#define OFFSET_LEN 9
int offset_int8[OFFSET_LEN] = {150, 200, 150, 200, 255, 200, 150, 200, 150};
int offset[OFFSET_LEN] = 
{
    -BUFFER_WIDTH-1,
    -BUFFER_WIDTH,
    -BUFFER_WIDTH+1,
    -1,
    0,
    1,
    BUFFER_WIDTH-1,
    BUFFER_WIDTH,
    BUFFER_WIDTH+1,
};

int button_hit(int mx, int my)
{
    return (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
}

void render_button(SDL_Renderer* r)
{
    SDL_SetRenderDrawColor(r, 0, 0, 255, 255);
    SDL_RenderFillRect(r, &(SDL_Rect){ b.x, b.y, b.w, b.h });
}

void render_clear(SDL_Renderer* r)
{
    SDL_SetRenderDrawColor(r, 0,0,0,255);
    SDL_RenderClear(r);
    render_button(r);
}

void button_clicked()
{
    for (int i = 0; i < BUFFER_SIZE; ++i)
        state[i] = 0;
}

void dump_buffer()
{
    for (int i = 0; i < BUFFER_SIZE; ++i)
    {
        printf("%d", state[i] == 0 ? 0 : 1);
        if ((i+1) % 28 == 0)
            printf("\n");
    }
}

void handle_drag(int mx, int my)
{ 
    if (mx < 0 || mx >= WINDOW_WIDTH || my < 0 || my >= WINDOW_HEIGHT)
        return;

    if (button_hit(mx, my))
        return;

    float buffer_x = ((float)mx / WINDOW_WIDTH) * 28.0f;
    float buffer_y = ((float)my / WINDOW_HEIGHT) * 28.0f;

    printf("buffer_x %f, buffer_y %f\n", buffer_x, buffer_y);

    int idx = (int)buffer_y * BUFFER_WIDTH + (int)buffer_x;
    if (idx < 0 || idx >= BUFFER_SIZE)
        return;

    for (int i = 0; i < OFFSET_LEN; ++i)
    {
        int curr = idx + offset[i];
        if (curr < 0 || curr >= BUFFER_SIZE)
            continue;
        state[curr] = min(state[curr] + offset_int8[i], 255);
    }
}

void render_state(SDL_Renderer* r)
{
    for (int i = 0; i < BUFFER_SIZE; ++i)
    {
        int x = i % BUFFER_WIDTH;
        int y = i / BUFFER_WIDTH;
        float rx = (float)x / 28.0f;
        float ry = (float)y / 28.0f;

        if (state[i])
        {
            SDL_SetRenderDrawColor(r, state[i], state[i], state[i], 255);
            SDL_RenderFillRect(r, &(SDL_Rect){rx * WINDOW_WIDTH, ry * WINDOW_HEIGHT, WINDOW_WIDTH / 28, WINDOW_HEIGHT / 28});
        }
    }
}

int main()
{
    SDL_Window* window;
    SDL_Renderer* renderer;
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
        while (SDL_PollEvent(&e))
        {
            SDL_GetMouseState(&mx, &my);

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
                        if (button_hit(mx, my)) 
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
            render_clear(renderer);
            render_state(renderer);
            SDL_RenderPresent(renderer);
            must_update = 0;
        }

        SDL_Delay(16); // ~60 FPS
    }

    cleanup_sdl(window, renderer);

    return 0;
}

