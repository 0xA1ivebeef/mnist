
make:
	gcc main.c -lSDL2 -lopenblas -lm -Wall -Wextra -o main

clean: 
	rm main

