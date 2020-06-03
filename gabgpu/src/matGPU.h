//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_MATGPU_H
#define GAB_MATGPU_H

#include <CL/cl.h>

typedef struct {
    cl_mem v;
    cl_int m, n, bytes;
    size_t lw_m, lw_n;
    size_t lw_m_only;


} Mat;

//Mat new_Mat(cl_context context, int m, int n, int *err);
Mat new_Mat(cl_context context, int m, int n,int *err);

void Mat_print(Mat *self, cl_command_queue queue);

void Mat_release(Mat *self);

size_t Mat_multiplo(int m, int max);

#endif //GAB_MATGPU_H
