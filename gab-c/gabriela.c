//
// Created by Xx220xX on 02/06/2020.include "gabriela.h"

#include "gabriela.h"
#include<stdlib.h> // calloc
#include<stdarg.h> // calloc
#include<string.h> // memcpy
#include<math.h>  // math functions
#include "stdio.h" // fwrite fread
#include"lcg.h" // randomic functions

#define REAL double
#define M_ij(m, i, j)m->v[i*m->n+j]
#define M_i(m, i)m->v[i]
#define LEN_DNN sizeof(struct _dnn)
#define LEN_MAT sizeof(struct _Mat)
#define LEN_REAL sizeof(REAL)

typedef struct _Mat {
    int m, n;
    REAL *v;
} *Mat, **pMat;

struct _dnn {
    int *arq;
    int *func_id;
    int L;
    double hitLearn;
    pMat w, a, b, z, dz, dw;
};

Mat newMat(int, int);

int releaseMat(Mat);

int createMatrix(DNN);

REAL ativa(int, REAL, ...);

REAL deriva(int, REAL, ...);

static LCG lcg = {
        0x5DEECE66DULL,
        11ULL,
        1ULL << 48,
        (1ULL << 48) - 1,
        0,
        1 << (sizeof(int) * 8 - 1)
};

/**
 *  cria uma rede neural com a funcao de ativação fixa para todas camadas
 *
 * @param arquitetura representa a quantidade de neuronios que cada camada conterá
 * @param la tamanho do vetor arquitetura
 * @param hitLearn taxa de aprendizado
 * @return uma rede neural pronta para uso
 */
DNN newDnn(int *arquitetura, int la, double hitLearn) {
    struct _dnn dnn = {0};
    DNN d = (DNN) calloc(1, LEN_DNN);
    int i;
    dnn.hitLearn = hitLearn;
    dnn.L = la - 1;
    dnn.arq = (int *) calloc(sizeof(int), la);
    memcpy(dnn.arq, arquitetura, sizeof(int) * la);
    createMatrix(&dnn);//cria as matrizes que serao usadas pela rede de acordo com a aquitetura
    la--;
    dnn.func_id = (int *) calloc(sizeof(int), la );
    for (i=0;i<la;i++)dnn.func_id[i] = TANH;
    memcpy(d, &dnn, LEN_DNN);
    return d;
}

DNN newDnnWithFunctions(int *arquitetura, int la, int *f_id,  double hitlearn) {
    DNN d = newDnn(arquitetura, la, hitlearn);
    memcpy(d->func_id, f_id, (la-1) * sizeof(int));
    return d;
}

/**
 * Libera os recursos alocados pela rede
 * @param d
 * @return
 */
int releaseDNN(DNN d) {
    int l;
    if (d) {
        if (d->arq)
            free(d->arq);
        releaseMat(d->a[0]);
        for (l = 1; l <= d->L; ++l) {
            releaseMat(d->w[l]);
            releaseMat(d->a[l]);
            releaseMat(d->b[l]);
            releaseMat(d->z[l]);
            releaseMat(d->dz[l]);
            releaseMat(d->dw[l]);
        }
        free(d->w);
        free(d->a);
        free(d->b);
        free(d->z);
        free(d->dz);
        free(d->dw);
        free(d->func_id);
        free(d);

        return 0;
    }
    return -1;
}


/**
 * Faz o feed forwad da rede com a entrada inp
 * @param dnn
 * @param inp
 * @return
 */
int call(DNN dnn, double *inp) {
    int i, j, l;
    // copia os dados para a rede
    if(!dnn)return -1;
    for (i = 0; i < dnn->a[0]->m; ++i)dnn->a[0]->v[i] = (REAL) inp[i];
    Mat w, a_, b, z, a;
    REAL soma;
    // as seguintes linhas faz os seguintes calculos
    //z[l] =  W[l]*a[l-1] + b[l] somatorio de pesos vezes entrada mais o bias
    //a[l] = f[l](z[l]) ativação do neuronio usando a função de ativação da camada
    for (l = 1; l <= dnn->L; ++l) {
        w = dnn->w[l];
        b = dnn->b[l];
        z = dnn->z[l];
        a = dnn->a[l];
        a_ = dnn->a[l - 1];
        REAL tmp;
        // tratando a softmax de maneira diferente
        for (i = 0; i < w->m; ++i) {
            tmp = b->v[i];
            for (int k = 0; k < a_->m; ++k) {
                tmp += w->v[i * w->n + k] * a_->v[k];
            }
            z->v[i] = tmp;
            a->v[i] = ativa(dnn->func_id[l - 1], tmp);
        }
        if (dnn->func_id[l - 1] == SOFTMAX) {
            soma = 0;
            for (i = 0; i < a->m; ++i) soma += M_i(a, i);// somatorio e^ai
            for (i = 0; i < a->m; ++i) M_i(a, i) = M_i(a, i) / soma;// e^ai /(somatorio e^aj )
        }

    }
    return 0;
}


/**
 * Função backpropagation
 * @param dnn
 * @param target
 * @return
 */
int learn(DNN dnn, double *target) {
    if(!dnn)return -1;
    int i, j, L, l;
    Mat al_, al, dwl, dwl_u, dzl, dzl_u, wl_u, zl, bl;
    REAL tmp, hl;
    L = dnn->L;
    hl = dnn->hitLearn;

    al = dnn->a[L];
    dzl = dnn->dz[L];
    bl = dnn->b[L];

    // last Layer
    //dzL = aL - y
    for (i = 0; i < al->m; ++i) {
        M_i(dzl, i) = M_i(al, i) - (REAL) target[i];
        M_i(bl, i) = M_i(bl, i) - hl * M_i(dzl, i);
    }

    // dwL = dzl*al_t
    dwl = dnn->dw[L];
    al_ = dnn->a[L - 1];
    for (i = 0; i < dzl->m; ++i) {
        for (j = 0; j < al_->m; ++j)
            M_ij(dwl, i, j) = M_i(dzl, i) * M_i(al_, j);
    }
    for (l = L - 1; l >= 1; l--) {
        dzl = dnn->dz[l];
        wl_u = dnn->w[l + 1];
        dzl_u = dnn->dz[l + 1];
        zl = dnn->z[l];
        al = dnn->a[l];
        dwl_u = dnn->dw[l + 1];
        // dzl = wl+t * dzl+  ** df(zl)
        for (i = 0; i < wl_u->n; ++i) {
            tmp = 0;
            for (j = 0; j < dzl_u->m; ++j) {
                tmp += M_ij(wl_u, j, i) * M_i(dzl_u, j);
                M_ij(wl_u, j, i) = M_ij(wl_u, j, i) - hl * M_ij(dwl_u, j, i);
            }
            tmp = tmp * deriva(dnn->func_id[l - 1], M_i(zl, i), M_i(al, i));
            M_i(dzl, i) = tmp;
        }
        // dwL = dzl*al_t
        dwl = dnn->dw[l];
        al_ = dnn->a[l - 1];
        for (i = 0; i < dzl->m; ++i)
            for (j = 0; j < al_->m; ++j)
                M_ij(dwl, i, j) = M_i(dzl, i) * M_i(al_, j);

    }

    wl_u = dnn->w[1];
    dwl = dnn->dw[1];

    for (i = 0; i < dwl->m * dwl->n; ++i) {
        M_i(wl_u, i) = M_i(wl_u, i) - hl * M_i(dwl, i);
    }
    return 0;
}

int getOut(DNN dnn, double *o) {
    int i;
    Mat mo = dnn->a[dnn->L];
    for (i = 0; i < mo->m; ++i) {
        o[i] = (double) mo->v[i];
    }
    return 0;
}

int setSeed(long long int seed) {
    lcg.atual = seed;
    return 0;
}

REAL rando(double max, double min) {
    max = max - min;
    return ((REAL) LCG_randD(&lcg)) * max + min;
}

/**
 *
 * @param dnn
 * @return
 *
 */
int randomize(DNN dnn) {
    int i, k;
    Mat w, b;
    if(!dnn)return -1;
    // epsilon = sqrt(6/(tam_camadas[i] + tam_camadas[i+1])) utilizado para inicializacao
    REAL epsilon;
    for (i = 1; i <= dnn->L; ++i) {
        w = dnn->w[i];
        b = dnn->b[i];
        epsilon = sqrt(6.0 / (w->m + w->n + 1));
        for (k = 0; k < w->m * w->n; ++k) {
            w->v[k] = rando(epsilon, -epsilon);
        }
        for (k = 0; k < b->m; ++k) {
            b->v[k] = rando(epsilon, -epsilon);
        }

    }
    return 0;
}


#define CABECALHO "NEURAL NETWORK\n"\
                  "LENGTH OF ARCHITECTURE = 4 BYTES\n"\
                  "ARQUITETURE = 4 * LENGTH_ARCHITECTURE BYTES\n"\
                  "FUNCTION ID = LENGTH_ARCHITECTURE - 1 BYTES\n"\
                  "HITE LEARN = 8 BYTES\n"\
                  "WEIGHTS WITH DIMENSION DEFINED BY ARCHITECTURE\n"

/**
 * Salva em binario os dados da rede
 * os primeiros 210 BYTES sao de cabecalho
 * @param d
 * @param file_name
 * @return
 */
int saveDNN(DNN d, char *file_name) {
    if (!d)return -2;
    int length = d->L + 1;
    int i, j, l;
    FILE *f = fopen(file_name, "wb");
    if (!f)return -1;

    fwrite(CABECALHO, sizeof(char), 210, f);
    fwrite(&length, sizeof(int), 1, f);
    fwrite(d->arq, length, sizeof(int), f);
    fwrite(d->func_id, length - 1, sizeof(int), f);
    fwrite(&d->hitLearn, 1, LEN_REAL, f);

    Mat b, w;
    for (l = 1; l <= d->L; l++) {
        b = d->b[l];
        w = d->w[l];
        for (i = 0; i < w->m; ++i) {
            fwrite(&M_i(b, i), 1, LEN_REAL, f);
            fwrite(w->v + (i * w->n), w->n, LEN_REAL, f);
        }
    }
    fclose(f);
    return 0;
}

DNN loadDNN(char *file_name) {
    FILE *f = fopen(file_name, "rb");
    DNN d;
    char cabecalho[211] = {0};
    int i, j, l;
    int length, *arq;
    if (!f)return NULL;
    fread(cabecalho, sizeof(char), 210, f);
    if (strcmp(cabecalho, CABECALHO)) return NULL;

    fread(&length, sizeof(int), 1, f);
    arq = calloc(length, sizeof(int));
    fread(arq, length, sizeof(int), f);
    d = newDnn(arq, length, .1);
    free(arq);

    fread(d->func_id, length - 1, sizeof(int), f);
    fread(&d->hitLearn, 1, LEN_REAL, f);
    Mat b, w;
    for (l = 1; l <= d->L; l++) {
        b = d->b[l];
        w = d->w[l];
        for (i = 0; i < w->m; ++i) {
            fread(&M_i(b, i), 1, LEN_REAL, f);
            fread(w->v + (i * w->n), w->n, LEN_REAL, f);
        }
    }
    fclose(f);
    return d;
}

int setFuncID(DNN self,int camada, int f_ID) {
    if (f_ID <0 || f_ID>5)return -1;
    camada--;
    if(camada<1 || camada>self->L) return -2;

    self->func_id[camada-1] = f_ID;
    return 0;
}

void printDNN(DNN dnn) {
    if(!dnn)return;
    int l;
    printf("camadas: %d\n",dnn->L+1);
    printf("Arquitetura: ");
    printf("(%d",dnn->arq[0]);
    for (l=1;l <=dnn->L;l++)printf(" %d",dnn->arq[l]);
    printf(")\n");

    for (l = 1; l <= dnn->L; ++l) {
        printf("camada %d, funcao de ativacao:  ",l+1);
        switch(dnn->func_id[l-1]) {
            case EXPERIMENTAL:
                printf("EXPERIMENTAL\n\tf(x) =  x E (0.5,inf)  -> ln(x+1.14), x E(-inf,-0.5) -> -ln(-x + 1.14), x E [-0.5,0.5] tanh(x)");
                break;
            case IDENTIDADE:
                printf("IDENTIDADE\n\tf(x) = x");
                break;
            case SIGMOID:
                printf("SIGMOIDE\n\tf(x) = 1/(1 + e^-x)");
                break;
            case TANH:
                printf("TANH\n\tf(x) = tanh(x)");
                break;
            case RELU:
                printf("RELU\n\tf(x) = x E(-inf,0) -> 0, x E (0,inf) -> x, x = 0 -> 0.5");
                break;
            case SOFTMAX:
                printf("SOFTMAX\n\tf(x) = e^x/(k + e^x) , onde k + e^x = sum(e^a[j])");
                break;
            default:
                fprintf(stderr, "WARNING: INVALID FUNCTION ID\n");
                break;
        }
        printf("\n");
    }
}


Mat newMat(int m, int n) {
    struct _Mat mt;
    mt.m = m;
    mt.n = n;
    mt.v = (REAL *) calloc(m * n, sizeof(REAL));
    Mat ans = (Mat) calloc(1, LEN_MAT);
    memcpy(ans, &mt, LEN_MAT);
    return ans;
}

int releaseMat(Mat mat) {
    if (mat) {
        if (mat->v) {
            free(mat->v);
        }
        free(mat);
    }
    return 0;
}

int createMatrix(DNN dnn) {
    int l;
    char nm[12] = {0};
    dnn->w = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->a = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->z = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->b = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->dw = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->dz = (pMat) calloc(dnn->L + 1, sizeof(Mat));
    dnn->a[0] = newMat(dnn->arq[0], 1);

    for (l = 1; l <= dnn->L; ++l) {
        dnn->w[l] = newMat(dnn->arq[l], dnn->arq[l - 1]);
        dnn->a[l] = newMat(dnn->arq[l], 1);
        dnn->b[l] = newMat(dnn->arq[l], 1);
        dnn->z[l] = newMat(dnn->arq[l], 1);
        dnn->dz[l] = newMat(dnn->arq[l], 1);
        dnn->dw[l] = newMat(dnn->arq[l], dnn->arq[l - 1]);

    }
    return 0;
}

REAL ativa(int f_id, REAL x, ...) {
    switch (f_id) {
        case EXPERIMENTAL:
            if (x > 0.5)return log(x + 1.14);
            if (x < 0.5)return -log(-x + 1.14);
            return tanh(x);
        case IDENTIDADE:
            return x;
        case SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case TANH:
            return tanh(x);
        case RELU:
            return x < 0 ? 0 : x;
        case SOFTMAX:
            return exp(x);
        default:
            fprintf(stderr, "WARNING: INVALID FUNCTION ID\n");
            return 0;
    }
}

REAL deriva(int f_id, REAL x, ...) {
    REAL tmp;
    va_list vars;
    switch (f_id) {
        case EXPERIMENTAL:
            if (x > 0.5)return 1.0 / (x + 1.14);
            if (x < 0.5)return 1.0 / (-x + 1.14);
            va_start(vars, x);
            tmp = va_arg(vars, REAL);
            tmp = 1.0 - (tmp * tmp);//1 - tanh^2
            va_end(vars);
            return tmp;
        case IDENTIDADE:
            return 1;
        case SIGMOID:
            va_start(vars, x);
            tmp = va_arg(vars, REAL);
            tmp = tmp * (1.0 - tmp);//sig/(1 - sig)^2
            va_end(vars);
            return tmp;
        case TANH:
            va_start(vars, x);
            tmp = va_arg(vars, REAL);
            tmp = 1.0 - (tmp * tmp);//1 - tanh^2
            va_end(vars);
            return tmp;
        case RELU:
            return x == 0 ? 0.5 : (x < 0 ? 0 : 1);
        case SOFTMAX:
            va_start(vars, x);
            tmp = va_arg(vars, REAL);
            tmp = tmp * (1 - tmp);//f(x) * (1  - f(x) )
            va_end(vars);
            return tmp;
        default:
            fprintf(stderr, "WARNING: INVALID FUNCTION ID\n");
            return 0;
    }
}
