#ifndef GABRIELA_H
#define GABRIELA_H
// FUNCOES DE ATIVAÇÃO
//identidade - 1, sigmoide - 2, tanh - 3, relu - 4, softmax - 5)
#define EXPERIMENTAL 0 //ALAN
#define IDENTIDADE 1
#define SIGMOID 2
#define TANH 3
#define RELU 4
#define SOFTMAX 5

typedef struct _dnn *DNN;

DNN newDnn(int *, int, double);

DNN newDnnWithFunctions(int *, int, int *, double);

int releaseDNN(DNN);

int call(DNN, double *);

int learn(DNN, double *);

int setSeed(long long int);

int randomize(DNN);

int getOut(DNN, double *);

int saveDNN(DNN, char *);

DNN loadDNN(char *);

int setFuncID(DNN, int, int);

void printDNN(DNN);

#endif