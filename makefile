
make:
	gcc main.c -lSDL2 -lopenblas -lm -o main

clean: 
	rm main

