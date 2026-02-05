
#define UI_IMPLEMENTATION
#include "ui.h"

#include <stdio.h>

typedef struct Button
{
    int x, y;
    int w, h;
} Button;

void save_state()
{
    // TODO implement
}

void render_button(SDL_Renderer* r, Button* b)
{
    SDL_SetRenderDrawColor(r, 0, 0, 255, 255);
    SDL_RenderFillRect(r, &(SDL_Rect){ b->x, b->y, b->w, b->h });
}

void clear_canvas(SDL_Renderer* r, Button* b)
{
    SDL_SetRenderDrawColor(r, 0,0,0,255);
    SDL_RenderClear(r);
    render_button(r, b);
    SDL_RenderPresent(r);
}

int main()
{
    SDL_Window* window;
    SDL_Renderer* renderer;
    init_sdl(&window, &renderer);

    Button button = {0, 0, 200, 50};
    render_button(renderer, &button);

    int running = 1;
    SDL_Event e;
    int mouse_x, mouse_y;
    int drag = 0;
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
                        if (mouse_x >= button.x && mouse_x <= button.x + button.w &&
                            mouse_y >= button.y && mouse_y <= button.y + button.h) 
                            save_state();
                            // clear_canvas(renderer, &button);
                            
                    }
                    printf("mouse clicked %d, %d\n", mouse_x, mouse_y);
                    drag = 1;
                    break;
                case SDL_MOUSEBUTTONUP:
                    drag = 0;
                    printf("mouse released %d, %d\n", mouse_x, mouse_y);
                    break;
            }
        }
        
        SDL_GetMouseState(&mouse_x, &mouse_y);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        if (drag)
            SDL_RenderFillRect(renderer, &(SDL_Rect) { mouse_x-10, mouse_y-10, 20, 20 } );

        SDL_RenderPresent(renderer);
        SDL_Delay(16); // ~60 FPS
    }

    cleanup_sdl(window, renderer);

    return 0;
}
