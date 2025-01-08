#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>



void genRandomMatrix(float* A, int M, int N);
void FillMatrix(float* A, float num, int M, int N);
void genFixedMatrix(float* A, int M, int N);
void copyMatrix(float* des, float* src, int M, int N);
#endif