@echo compile for 32 bits
gcc -m32 -c gabriela.c -o tmp.o
gcc -m32 -shared tmp.o -o lib/libgab32.dll
del /f tmp.o
@echo compile for 64 bits
gcc -m64 -c gabriela.c -o tmp.o
gcc -m64 -shared tmp.o -o lib/libgab64.dll
del /f tmp.o
