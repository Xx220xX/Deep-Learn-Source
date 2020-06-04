#ifndef GABRIELA_H
#define GABRIELA_H


typedef struct _dnn * DNN;

DNN newDnn(int *, int, double);

int releaseDNN(DNN);

int call(DNN, double *);

int learn(DNN, double *);

int randomize(DNN);

int setSeed(long long int);

int getOut(DNN, double *);


#endif