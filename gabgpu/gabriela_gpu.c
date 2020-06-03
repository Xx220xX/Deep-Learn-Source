#include"gabriela_gpu.h"
#include <time.h>
#include "src/gabriela.h"


static WrapperCL wpcl;
static int gpu_init = 0;

void setworkSizes(int maxWorks) {
    gab_set_max(maxWorks);
}

void teste() {
    printf("c is okay\n");
}

void initGPU(const char *src) {
    if (gpu_init)return;
    WrapperCL_init(&wpcl, src);
    intern_setSeed(time(0));
    size_t maxLW = 1;
    int error = clGetDeviceInfo(wpcl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
    if (error)fprintf(stderr, "falha ao checar valor error id: %d\n", error);
    gab_set_max(maxLW);
    gpu_init = 1;

}

void initWithFile(const char *filename) {
    FILE *f;
    f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "arquivo nao encontrado no caminho %s\n", filename);
        return;
    }
    char *src = 0;
    long int size = 0;
    fseek(f, 0L, SEEK_END);
    size = ftell(f);
    src = calloc(size, sizeof(char));
    fseek(f, 0L, SEEK_SET);
    fread(src, sizeof(char), size, f);
    src[size - 1] = 0;
    initGPU(src);
    free(src);
}

void endGPU() {
    if (!gpu_init)return;
    gpu_init = 0;
    WrapperCL_release(&wpcl);
}

void call(Gab *p_gab, double *inp) {
    DNN_call(p_gab->gab, inp);
}

void learn(Gab *p_gab, double *trueOut) {
    DNN_learn(p_gab->gab, trueOut);
}

void release(Gab *p_gab) {
    DNN_release(p_gab->gab);
    free(p_gab->gab);
}

int create_DNN(Gab *p_gab, int *arq, int l_arq, double hitLean) {
    p_gab->size = sizeof(DNN);
    int error = 0;
    DNN *dnn = calloc(1, p_gab->size);
    *dnn = new_DNN(&wpcl, arq, l_arq, hitLean, &error);
    p_gab->gab = dnn;
    if (error) {
        fprintf(stderr, "falha ao criar rede neural\n");
        free(dnn);
        return error;
    }
    return 0;
}

void getoutput(Gab *p_gab, double *out) {
    DNN *gab = (DNN *) p_gab->gab;
    memcpy(out, gab->out, gab->n[gab->L] * sizeof(double));
}

void setSeed(unsigned long int seed) {
    intern_setSeed(seed);
}

void checkLW() {
    size_t maxLW = 1;
    int error = clGetDeviceInfo(wpcl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
    if (error)fprintf(stderr, "falha ao checar valor error id: %d\n", error);
    else printf("max work group size: %ld\n", maxLW);

}

int randomize(Gab *p_gab) {
    int error = 0;
    DNN_randomize(p_gab->gab, &error);
    return error;
}

#define op(a, b) ((int )a) ^ ((int )b)

void testXor(char *file) {

    initWithFile(file);
    Gab gab;
    setSeed(1);
    int arquitetura[] = {2, 256, 128, 64, 32, 16, 8, 4, 2, 1};
    int la = sizeof(arquitetura) / sizeof(int);
    create_DNN(&gab, arquitetura, la, 0.1);
    double input[][2] = {{1, 1},
                         {0, 1},
                         {1, 0},
                         {0, 0}};
    double out[4][1];
    for (int i = 0; i < 4; ++i) out[i][0] = op(input[i][0], input[i][1]);
    int epocas = 0;
    int maxEpocas = 100;
    int i;
    double o = 0.0;
    double minEnergia = 1e-3;
    double energia = minEnergia + 1;
    clock_t t0 = clock();

    for (epocas = 0; energia > minEnergia && epocas < maxEpocas; epocas++) {
        for (i = 0; i < 4; ++i) {
            call(&gab, input[i]);
            learn(&gab, out[i]);
            getoutput(&gab, &o);
            energia += (o - out[i][0]) * (o - out[i][0]);
        }
        energia /= 2.0;
    }

    t0 = clock() - t0;
    printf("depois de %d epocas a energia foi %lf\n", epocas, energia);

    printf("tempo total gasto %ld ms\n", t0);
    printf("tempo medio gasto por exemplo %lf ms\n", t0 / ((double) epocas * 4.0));

    endGPU();
}

