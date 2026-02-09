
#ifndef UI_H
#define UI_H

#include <SDL2/SDL.h>

int init_sdl(SDL_Window** w, SDL_Renderer** r);
void cleanup(SDL_Window* w, SDL_Renderer* r);

#ifdef UI_IMPLEMENTATION

#define WINDOW_WIDTH 840
#define WINDOW_HEIGHT 840
#define WINDOW_SIZE WINDOW_HEIGHT * WINDOW_WIDTH

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

#endif
#endif
