@echo compile for 32 bits
gcc -m32 -c gabriela.c  lcg.c  
gcc -m32 -shared *.o -o lib/libgab32.dll
del /f *.o
@echo compile for 64 bits
gcc -m64 -c -c gabriela.c  lcg.c  
gcc -m64 -shared *.o -o lib/libgab64.dll
del /f *.o
pause