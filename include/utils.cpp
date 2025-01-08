#include "utils.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>



void genRandomMatrix(float* A, int M, int N) {
    srand(time(NULL));   // Initialization, should only be called once.
    float a = 5.0;
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            A[i * N + j] = (float) rand() / ((float)RAND_MAX / a);
        }
    }
}

void FillMatrix(float* A, float num, int M, int N) {
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            A[i * N + j] = num;
        }
    }
}

void genFixedMatrix(float* A, int M, int N) {
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            if ( i >= M * N / 2) A[i * N + j] = 2;
            else {
                A[i * N + j] = 0;
            }
        }
    }
}



void copyMatrix(float* des, float* src, int M, int N) {
    for ( int i = 0; i < M; i ++ ) {
        for (int j = 0; j < N; j ++) {
            des[i * N + j] = src[i * N + j];
        }
    }
}