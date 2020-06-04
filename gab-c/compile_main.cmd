@echo compile main
gcc gabriela.c -o gabriela.o -c
gcc main.c -o main.o -c
gcc -o main main.o gabriela.o 
del /f main.o
del /f gabriela.o

