CC = gcc
RM = del /f
all: gabriela.c gabriela.h

gabriela.c : lib/lgab32.dll lib/lgab64.dll

lgab32.dll: tmp_lgab32.o
	$(CC) -m32 -shared tmp_lgab32.o -o lib/libgab32.dll
	$(RM) tmp_lgab32.o
lgab64.dll: tmp_lgab64.o
	$(CC) -m64 -shared tmp_lgab64.o -o lib/libgab64.dll
	$(RM) tmp_lgab64.o
tmp_lgab32.o:
	$(CC) -m32 -c gabriela.c  -o tmp_lgab32.o

tmp_lgab64.o:
	$(CC) -m64 -c gabriela.c  -o tmp_lgab64.o
