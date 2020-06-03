#include <stdio.h>
#include <time.h>
#include "gabriela_gpu.h"


int main() {
    testXor("kernel_src.cl");
    return 0;
}
