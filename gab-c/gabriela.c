//
// Created by Xx220xX on 02/06/2020.include "gabriela.h"

#include "gabriela.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "stdio.h"

#define REAL double
typedef struct _Mat {
    int m, n;
    REAL *v;
    char name[12];
} *Mat;

#define M_ij(m, i, j)m->v[i*m->n+j]
#define M_i(m, i)m->v[i]

typedef Mat *pMat;
struct _dnn {
    int *arq;
    int L;
    int functionID;
    double hitLearn;
    pMat w, a, b, z, dz, dw;
};
#define LEN_DNN sizeof(struct _dnn)
#define LEN_MAT sizeof(struct _Mat)

Mat newMat(int, int);

int releaseMat(Mat);

int createMatrix(DNN);

REAL ativa(int, REAL);

REAL deriva(int, REAL);

DNN newDnn(int *arquitetura, int la, double hitLearn) {
    struct _dnn dnn = {0};
    DNN d = (DNN) calloc(1, LEN_DNN);
    dnn.hitLearn = hitLearn;
    dnn.L = la - 1;
    dnn.arq = (int *) calloc(sizeof(int), la);
    memcpy(dnn.arq, arquitetura, sizeof(int) * la);
    createMatrix(&dnn);
    memcpy(d, &dnn, LEN_DNN);
    return d;
}

void printDNN(FILE *f, DNN d) {
    int i;
    fprintf(f, "Arquiterura: (");
    for (i = 0; i <= d->L; i++) {
        fprintf(f, "%d", d->arq[i]);
        if (i < d->L)fprintf(f, " ");
    }
    fprintf(f, ")\n");
    fprintf(f, "camada de entrada\n a_0 %dx%d\n", d->a[0]->m, d->a[0]->n);
    for (i = 1; i < d->L; i++) {
        fprintf(f, "\ncamada %d\n", i);
        fprintf(f, "\t%s: %dx%d\n", d->w[i]->name, d->w[i]->m, d->w[i]->n);
        fprintf(f, "\t%s: %dx%d\n", d->b[i]->name, d->b[i]->m, d->b[i]->n);
        fprintf(f, "\t%s: %dx%d\n", d->z[i]->name, d->z[i]->m, d->z[i]->n);
        fprintf(f, "\t%s: %dx%d\n", d->a[i]->name, d->a[i]->m, d->a[i]->n);
        fprintf(f, "\t%s: %dx%d\n", d->dz[i]->name, d->dz[i]->m, d->dz[i]->n);
        fprintf(f, "\t%s: %dx%d\n", d->dw[i]->name, d->dw[i]->m, d->dw[i]->n);
    }

}

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
        free(d);
        return 0;
    }
    return -1;
}


void printM(FILE *f, Mat mat, char *fmt,char *seplin, char *end) {
    fprintf(f, "%% %dx%d:\n%s = [", mat->m, mat->n, mat->name);
    int i, j;
    for (i = 0; i < mat->m; ++i) {
        for (j = 0; j < mat->n; ++j) {
            fprintf(f, fmt, mat->v[i * mat->n + j]);
        }
        if (i + 1 < mat->m)
            fprintf(f, "%s",seplin);
    }
    fprintf(f, "]%s", end);
}

int call(DNN dnn, double *inp) {
    int i = 0, j, l;
    for (i = 0; i < dnn->a[0]->m; ++i)dnn->a[0]->v[i] = (REAL) inp[i];
    Mat w, a_, b, z, a;
    for (l = 1; l <= dnn->L; ++l) {
        w = dnn->w[l];
        b = dnn->b[l];
        z = dnn->z[l];
        a = dnn->a[l];
        a_ = dnn->a[l - 1];
        REAL tmp;
        for (i = 0; i < w->m; ++i) {
            tmp = b->v[i];
            for (int k = 0; k < a_->m; ++k) {
                tmp += w->v[i * w->n + k] * a_->v[k];
            }
            z->v[i] = tmp;
            a->v[i] = ativa(dnn->functionID, tmp);
        }
    }
    return 0;
}



int learn(DNN dnn, double *inp) {
    FILE *f = fopen("../oc.m","w");
    fprintf(f,"clc\nclear all\ndisp('LEARN');\n");
    // vars
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
        M_i(dzl, i) = M_i(al, i) - (REAL) inp[i];
        M_i(bl, i) = M_i(bl, i) - hl * M_i(dzl, i);
    }

    // dwL = dzl*al_t
    fprintf(f,"disp('');\ndisp('dwL = dzl*al_t')\n");
    dwl = dnn->dw[L];
    al_ = dnn->a[L - 1];
    for (i = 0; i < dzl->m; ++i) {
        for (j = 0; j < al_->m; ++j)
            M_ij(dwl, i, j) = M_i(dzl, i) * M_i(al_, j);
    }
    printM(f,dzl,"%lf ",";","\n");
    printM(f,al_,"%lf ",";","\n");
    printM(f,dwl,"%lf ",";","\n");
    fprintf(f,"oct = %s*(%s')\n",dzl->name,al_->name);

    for (l = L - 1; l >= 1; l--) {
        dzl = dnn->dz[l];
        wl_u = dnn->w[l + 1];
        dzl_u = dnn->dz[l + 1];
        zl = dnn->z[l];
        dwl_u = dnn->dw[l + 1];
        // dzl = wl+t * dzl+  ** df(zl)
        fprintf(f,"disp('');\ndisp('dzl = wl+t * dzl+  ** df(zl)')\n");
        for (i = 0; i < wl_u->n; ++i) {
            tmp = 0;
            for (j = 0; j < dzl_u->m; ++j) {
                tmp += M_ij(wl_u, j, i) * M_i(dzl_u, j);
                M_ij(wl_u, j, i) = M_ij(wl_u, j, i) - hl * M_ij(dwl_u, j, i);
            }
            tmp = tmp * deriva(dnn->functionID, M_i(zl, i));
            M_i(dzl, i) = tmp;
        }
        printM(f,wl_u,"%lf ",";",";\n");
        printM(f,dzl_u,"%lf ",";",";\n");
        printM(f,zl,"%lf ",";",";\n");
        printM(f,dzl,"%lf ",";","\n");
        fprintf(f,"oct = ((%s') * %s) .* (1.0 ./ (cosh(%s).*cosh(%s)))\n\n",wl_u->name,dzl_u->name,zl->name,zl->name);
        fprintf(f,"disp('_______');\n");

        // dwL = dzl*al_t
        dwl = dnn->dw[l];
        al_ = dnn->a[l - 1];
        for (i = 0; i < dzl->m; ++i)
            for (j = 0; j < al_->m; ++j)
                M_ij(dwl, i, j) = M_i(dzl, i) * M_i(al_, j);

    }

    wl_u = dnn->w[1];
    dwl = dnn->w[1];

    for (i = 0; i < dwl->m * dwl->n; ++i) {
        M_i(wl_u, i) = M_i(wl_u, i) - hl * M_i(dwl, i);
    }
    fclose(f);
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
    srand(seed);
    return 0;
}

REAL rando() {
    return (REAL) rand() / RAND_MAX * 2.0L - 1.0;
}

int randomize(DNN dnn) {
    int i, k;
    Mat w, b;
    for (i = 1; i <= dnn->L; ++i) {
        w = dnn->w[i];
        b = dnn->b[i];
        for (k = 0; k < w->m * w->n; ++k) {
            w->v[k] = rando();
        }
        for (k = 0; k < b->m; ++k) {
            b->v[k] = rando();
        }

    }
    return 0;
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

#define MNAME(M, nameM, l)\
    snprintf(nm,12,"%s_%d",nameM,l);\
    memcpy(M->name,nm,12)

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
    MNAME(dnn->a[0], "a", 0);
    for (l = 1; l <= dnn->L; ++l) {
        dnn->w[l] = newMat(dnn->arq[l], dnn->arq[l - 1]);
        dnn->a[l] = newMat(dnn->arq[l], 1);
        dnn->b[l] = newMat(dnn->arq[l], 1);
        dnn->z[l] = newMat(dnn->arq[l], 1);
        dnn->dz[l] = newMat(dnn->arq[l], 1);
        dnn->dw[l] = newMat(dnn->arq[l], dnn->arq[l - 1]);

        MNAME(dnn->w[l], "w", l);
        MNAME(dnn->a[l], "a", l);
        MNAME(dnn->b[l], "b", l);
        MNAME(dnn->z[l], "z", l);
        MNAME(dnn->dz[l], "dz", l);
        MNAME(dnn->dw[l], "dw", l);
    }
    return 0;
}

REAL ativa(int f_id, REAL x) {
    return tanh(x);
}

REAL deriva(int f_id, REAL x) {
    REAL tmp = cosh(x);
    tmp =  (REAL) 1.0 / (tmp * tmp);
    return tmp;
}
