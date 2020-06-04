cmake_minimum_required(VERSION 3.15)
project(gabriela_clib C)

set(CMAKE_C_STANDARD 11)


#compile main.c 
add_executable(gabriela_clib main.c gabriela.h gabriela.c)

