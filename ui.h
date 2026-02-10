
#ifndef UI_H
#define UI_H

#include <SDL2/SDL.h>

int init_sdl(SDL_Window** w, SDL_Renderer** r);
void cleanup(SDL_Window* w, SDL_Renderer* r);

#ifdef UI_IMPLEMENTATION

typedef struct Button
{
    int x, y;
    int w, h;
} Button;

#define BUFFER_WIDTH 28
#define BUFFER_HEIGHT 28
#define BUFFER_SIZE BUFFER_WIDTH*BUFFER_HEIGHT

#define OFFSET_LEN 9
int offset_int8[OFFSET_LEN] = {50, 70, 50, 70, 120, 70, 50, 70, 50};
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

#define WINDOW_WIDTH 840
#define WINDOW_HEIGHT 840
#define WINDOW_SIZE WINDOW_HEIGHT * WINDOW_WIDTH

uint8_t state[BUFFER_SIZE] = {0};
Button b = {0, 0, 200, 50};

int init_sdl(SDL_Window** w, SDL_Renderer** r)
{
    SDL_Init(SDL_INIT_VIDEO);
    *w = SDL_CreateWindow("SDL Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!*w)
    { 
        SDL_Quit();
        return 1;
    }

    *r = SDL_CreateRenderer(*w, -1, SDL_RENDERER_ACCELERATED);
    if (!*r)
    { 
        SDL_DestroyWindow(*w);
        SDL_Quit();
        return 1;
    }

    return 0;
}

void cleanup_sdl(SDL_Window* w, SDL_Renderer* r)
{
    SDL_DestroyWindow(w);
    SDL_DestroyRenderer(r);
    SDL_Quit();
}

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

#endif
#endif
