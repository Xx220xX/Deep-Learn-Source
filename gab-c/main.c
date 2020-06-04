#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gabriela.h"

void printV(const char *m, double *v, int s) {
    int i;
    printf("%s ", m);
    for (i = 0; i < s; i++) {
        printf("%lf ", v[i]);
    }
    printf("\n");
}

int main(int nargs,char **args) {

    int max_epocas = 10;
    if(nargs>1){
	max_epocas = atoi(args[1]);
	if(max_epocas <1){
		printf("FALHA\n");
		max_epocas = 10;
	}

	printf("mudando max_epocas para %d\n",max_epocas);
    }
    int arq[] = {2, 3, 1};
    setSeed(time(0));
    DNN d = newDnn(arq, sizeof(arq) / sizeof(int), 0.1);
    randomize(d);

    double inp[4][2] = {{1.0, 1.0},{1.0, 0},{0,   1.0},{0,   0},};
    double target[4][1] = {0};
    int i, j;
    unsigned int v0,v1;
    for (i = 0; i < 4; ++i) {
        v0 = inp[i][0], v1 = inp[i][1];
        target[i][0] = v0 ^ v1;
    }
    double out[1];
    double  energia;
    for (i = 0; i < max_epocas; ++i) {
        energia = 0;
        for (j = 0; j < 4; ++j) {
            call(d, inp[j]);
            getOut(d, out);
            energia += (out[0] - target[j][0])*(out[0] - target[j][0]);
//            printf("%lf xor %lf = %lf | %lf \n", inp[j][0], inp[j][1], out[0], target[j][0]);
            learn(d, target[j]);
        }
        energia/=2.0;
        printf("epoca %d energia  %lf\n",i+1,energia);
//       printf("\n\n\n");
    }
    releaseDNN(d);
    system("pause");
    return 0;
}
