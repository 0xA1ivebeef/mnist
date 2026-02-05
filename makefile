
make:
	gcc main.c -lSDL2 -lcblas -lm -o main

clean: 
	rm main

