#include <stdio.h>
#include "gabriela.h"

void printV(const char *m, double *v, int s) {
    int i;
    printf("%s ", m);
    for (i = 0; i < s; i++) {
        printf("%lf ", v[i]);
    }
    printf("\n");
}

int main() {
    int arq[] = {2, 3, 1};
    DNN d = newDnn(arq, sizeof(arq) / sizeof(int), 0.1);
//    printW(d,1);
    setSeed(1LL);
    randomize(d);
//    printW(d,1);
    double inp[4][2] = {{1.0, 1.0},
                        {1.0, 0},
                        {0,   1.0},
                        {0,   0},};
    double target[4][1] = {0};
    int i, j;
    for (i = 0; i < 4; ++i) {
        printf("[%lf %lf]\n", inp[i][0], inp[i][1]);
    }
    for (i = 0; i < 4; ++i) {
        unsigned int v0 = inp[i][0], v1 = inp[i][1];
        target[i][0] = v0 ^ v1;
    }
    double out[1];
    double  energia;
    for (i = 0; i < 1; ++i) {
        energia = 0;
        for (j = 0; j < 1; ++j) {
            call(d, inp[j]);
            getOut(d, out);
            energia += (out[0] - target[j][0])*(out[0] - target[j][0]);
//            printf("%lf xor %lf = %lf | %lf \n", inp[j][0], inp[j][1], out[0], target[j][0]);
            learn(d, target[j]);
        }
        energia/=2.0;
        printf("%lf\n",energia);
//       printf("\n\n\n");
    }
    printf("okay all\n");
    releaseDNN(d);
    return 0;
}
