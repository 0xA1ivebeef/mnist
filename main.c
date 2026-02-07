
#define UI_IMPLEMENTATION
#include "ui.h"

#include <stdio.h>

typedef struct Button
{
    int x, y;
    int w, h;
} Button;

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WINDOW_SIZE WINDOW_HEIGHT * WINDOW_WIDTH

#define BUFFER_WIDTH 28
#define BUFFER_HEIGHT 28
#define BUFFER_SIZE BUFFER_WIDTH*BUFFER_HEIGHT
uint8_t state[BUFFER_SIZE] = {0};


#define OFFSET_LEN 9
int offset_int8[OFFSET_LEN] = {150, 200, 150, 200, 255, 200, 150, 200, 150};
int offset[OFFSET_LEN] = 
{
    // nw, n, ne, w, self, e, sw, s, se
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

void render_button(SDL_Renderer* r, Button* b)
{
    SDL_SetRenderDrawColor(r, 0, 0, 255, 255);
    SDL_RenderFillRect(r, &(SDL_Rect){ b->x, b->y, b->w, b->h });
}

void render_clear(SDL_Renderer* r, Button* b)
{
    SDL_SetRenderDrawColor(r, 0,0,0,255);
    SDL_RenderClear(r);
    render_button(r, b);
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
        printf("%d ", state[i]);
        if ((i+1) % BUFFER_WIDTH == 0)
            printf("\n");
    }
    printf("\n");
}

void handle_drag(int mx, int my)
{ 
    if (mx < 0 || mx >= WINDOW_WIDTH || my < 0 || my >= WINDOW_HEIGHT)
        return;

    // 0..WINDOW_WIDTH -> 0..BUFFER_SIZE
    float mouse_x = (float)mx / WINDOW_WIDTH * 28;
    float mouse_y = (float)my / WINDOW_HEIGHT * 28;

    int idx = mouse_y * BUFFER_WIDTH + mouse_x;
    if (idx < 0 || idx > BUFFER_SIZE)
        return;

    for (int i = 0; i < OFFSET_LEN; ++i)
    {
        int curr = idx + offset[i];
        if (curr >= 0 && curr < BUFFER_SIZE)
            state[curr] = offset_int8[i];
    }
}

void render_state(SDL_Renderer* r)
{
    SDL_SetRenderDrawColor(r, 255, 255, 255, 255);
    for (int i = 0; i < BUFFER_SIZE; ++i)
    {
        int x = i % BUFFER_WIDTH;
        int y = i / BUFFER_WIDTH;
        float dx = x / 28;
        float dy = y / 28 * WINDOW_HEIGHT;
        if (state[i])
            SDL_RenderDrawRect(r, &(SDL_Rect){dx*WINDOW_WIDTH, dy*WINDOW_HEIGHT, WINDOW_WIDTH / 28, WINDOW_HEIGHT / 28});
    }
}

int main()
{
    SDL_Window* window;
    SDL_Renderer* renderer;
    init_sdl(&window, &renderer);

    Button b = {0, 0, 200, 50};

    int running = 1;
    SDL_Event e;
    int mx, my;
    int drag = 0;
    int must_update = 0;

    render_clear(renderer, &b);
    SDL_RenderPresent(renderer);

    while (running)
    {
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
                        if (mx >= b.x && mx <= b.x + b.w &&
                            my >= b.y && my <= b.y + b.h) 
                        {
                            button_clicked();
                            must_update = 1;
                        }
                    }
                    printf("mouse clicked %d, %d\n", mx, my);
                    drag = 1;
                    break;
                case SDL_MOUSEBUTTONUP:
                    drag = 0;
                    printf("mouse released %d, %d\n", mx, my);
                    break;
            }
        }
        
        SDL_GetMouseState(&mx, &my);

        if (drag)
        {
            handle_drag(mx, my);
            must_update = 1;
        }

        if (must_update)
        {
            dump_buffer();
            printf("updating\n");
            render_clear(renderer, &b);
            render_state(renderer);
            SDL_RenderPresent(renderer);
            must_update = 0;
        }

        SDL_Delay(16); // ~60 FPS
    }

    cleanup_sdl(window, renderer);

    return 0;
}

