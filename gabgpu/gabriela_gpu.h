//
// Created by Xx220xX on 07/05/2020.
//

typedef struct {
    void *gab;
    unsigned int size;
} Gab;

void teste();

int create_DNN(Gab *p_gab, int *arq, int l_arq, double hitLean);

void initGPU(const char *src);

void initWithFile(const char *filename);

void endGPU();

void setworkSizes(int maxWorks);

void call(Gab *p_gab, double *inp);

void learn(Gab *p_gab, double *trueOut);

void release(Gab *p_gab);

void getoutput(Gab *p_gab, double *out);

void setSeed(unsigned long int seed);

int randomize(Gab *p_gab);

void checkLW();
void testXor(char *file);