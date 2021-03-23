#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>

#define MAX 10
#define SIZE 64

__global__ void init(unsigned int seed, curandState_t *states, int offset) {
    curand_init(seed, blockIdx.x, offset, &states[blockIdx.x]);
}

__global__ void randoms(curandState_t *states, float *numbers) {
  numbers[blockIdx.x] = (float) (curand(&states[blockIdx.x]) % MAX);
}

void gen_rand(float *B, float *dev_B) {
    curandState_t *states;
    cudaMalloc((void**) &states, SIZE * sizeof(curandState_t));

    init<<<SIZE, 1>>>(time(NULL), states, 0);
    randoms<<<SIZE, 1>>>(states, dev_B);
    cudaMemcpy(B, dev_B, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(states);
}

void run_cublas_calcs(float *A, float *B, float *C, float *dev_A, float *dev_B) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("cuBLAS: \n");
    printf("Vector A is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", A[i]);
    }
    printf("\nVector B is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", B[i]);
    }

    int *idx = (int*)malloc(sizeof(int));
    cublasIsamax(handle, SIZE, dev_B, 1, idx);
    printf("\n\nThe smallest index of the element with max magnitude of Vector B is: %d\n",
           *idx - 1);
    printf("The value of Vector B at index %d is %f\n", *idx - 1, B[*idx - 1]);

    cublasIsamin(handle, SIZE, dev_B, 1, idx);
    printf("\nThe smallest index of the element with min magnitude of Vector B is: %d\n",
           *idx - 1);
    printf("The value of Vector B at index %d is %f\n", *idx - 1, B[*idx - 1]);

    cublasSasum(handle, SIZE, dev_B, 1, C);
    printf("\nThe sum of Vector B is: %f\n", C[0]);

    cublasSdot(handle, SIZE, dev_A, 1, dev_B, 1, C);
    printf("\nThe dot product of Vector A and B is: %f\n\n", C[0]);

    cublasDestroy(handle);
    free(idx);
}

void cusparse_saxpyi(cusparseHandle_t handle, int nnz, float *C, float *dev_B_val,
                     int *dev_B_idx, float *dev_A) {
    float alpha = 1;
    cusparseSaxpyi(handle, nnz, &alpha, dev_B_val, dev_B_idx, dev_A, CUSPARSE_INDEX_BASE_ZERO);
    cublasGetVector(SIZE, sizeof(float), dev_A, 1, C, 1);
    printf("\n\nVector A plus Vector B is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", C[i]);
    }
}

void cusparse_gather(cusparseHandle_t handle, int nnz, float *A, float *B, float *C,
                     float *dev_A, float *dev_B_val, int *dev_B_idx, int *B_idx) {
    cublasSetVector(SIZE, sizeof(float), A, 1, dev_A, 1);
    cusparseSgthr(handle, nnz, dev_A, dev_B_val, dev_B_idx, CUSPARSE_INDEX_BASE_ZERO);
    cublasGetVector(nnz, sizeof(float), dev_B_val, 1, C, 1);
    printf("\n\nVector A gathered into Vector B is: ");
    int nnz_counter = 0;
    for (int i = 0; i < SIZE; i++) {
        if (i == B_idx[nnz_counter]){
            printf("%f ", C[nnz_counter]);
            nnz_counter++;
        }
        else
            printf("%f ", B[i]);
    }
}

void cusparse_scatter(cusparseHandle_t handle, int nnz, float *C, float *B_val,
                      float *dev_A, float *dev_B_val, int *dev_B_idx) {
    cublasSetVector(nnz, sizeof(float), B_val, 1, dev_B_val, 1);
    cusparseSsctr(handle, nnz, dev_B_val, dev_B_idx, dev_A, CUSPARSE_INDEX_BASE_ZERO);
    cublasGetVector(SIZE, sizeof(float), dev_A, 1, C, 1);
    printf("\n\nVector B scattered into Vector A is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", C[i]);
    }
}

void cusparse_rotate(cusparseHandle_t handle, int nnz, float *A, float *B, float *C,
                     float *dev_A, float *dev_B_val, int *dev_B_idx, int *B_idx) {
    cublasSetVector(SIZE, sizeof(float), A, 1, dev_A, 1);
    float c_coeff = 0.5;
    float s_coeff = 0.866025;
    cusparseSroti(handle, nnz, dev_B_val, dev_B_idx, dev_A, &c_coeff, &s_coeff, CUSPARSE_INDEX_BASE_ZERO);
    cublasGetVector(SIZE, sizeof(float), dev_A, 1, C, 1);
    printf("\n\nThe result of the Givens rotation on Vector A and B is: \n");
    printf("Vector A: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", C[i]);
    }
    cublasGetVector(nnz, sizeof(float), dev_B_val, 1, C, 1);
    int nnz_counter = 0;
    for (int i = 0; i < SIZE; i++) {
        if (i == B_idx[nnz_counter]){
            printf("%f ", C[nnz_counter]);
            nnz_counter++;
        }
        else
            printf("%f ", B[i]);
    }
}

void run_cusparse_calcs(float *A, float *B, float *C, float *dev_A, float *dev_B) {
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    int nnz = 0;
    float *B_val = (float*)malloc(SIZE * sizeof(float));
    int *B_idx = (int*)malloc(SIZE * sizeof(int));

    printf("cuSPARSE: \n");
    printf("Vector A is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", A[i]);
    }
    printf("\nVector B is: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%f ", B[i]);
        if(B[i] != 0) {
            B_idx[nnz] = i;
            B_val[nnz] = B[i];
            nnz++;
        }
    }
    
    int *dev_B_idx;
    float *dev_B_val;
    cudaMalloc(&dev_B_idx, nnz * sizeof(int));
    cudaMalloc(&dev_B_val, nnz * sizeof(float));

    cublasSetVector(nnz, sizeof(int), B_idx, 1, dev_B_idx, 1);
    cublasSetVector(nnz, sizeof(float), B_val, 1, dev_B_val, 1);

    cusparse_saxpyi(handle, nnz, C, dev_B_val, dev_B_idx, dev_A);
    cusparse_gather(handle, nnz, A, B, C, dev_A, dev_B_val, dev_B_idx, B_idx);
    cusparse_scatter(handle, nnz, C, B_val, dev_A, dev_B_val, dev_B_idx);
    cusparse_rotate(handle, nnz, A, B, C, dev_A, dev_B_val, dev_B_idx, B_idx);

    free(B_idx);
    free(B_val);
    cudaFree(dev_B_idx);
    cudaFree(dev_B_val);
}

double main_pageable() {
    printf("Pageable Memory: \n");
    float *dev_A, *dev_B;
    float *A = (float*)malloc(SIZE * sizeof(float));
    float *B = (float*)malloc(SIZE * sizeof(float));
    float *C = (float*)malloc(SIZE * sizeof(float));

    cudaMalloc(&dev_A, SIZE * sizeof(float));
    cudaMalloc(&dev_B, SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        A[i] = (float) i;
    }
    gen_rand(B, dev_B);

    cublasSetVector(SIZE, sizeof(float), A, 1, dev_A, 1);
    cublasSetVector(SIZE, sizeof(float), B, 1, dev_B, 1);

    clock_t start = clock();
    run_cublas_calcs(A, B, C, dev_A, dev_B);

    for (int i = 0; i < SIZE; i++) {
        if (i < SIZE - 1)
            B[i] = 0;
    }

    run_cusparse_calcs(A, B, C, dev_A, dev_B);
    double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    free(A);
    free(B);
    free(C);
    cudaFree(dev_A);
    cudaFree(dev_B);

    return pageable_time;
}

double main_pinned() {
    printf("Pinned Memory: \n");
    float *A, *B, *C, *dev_A, *dev_B, *dev_C;
	cudaMallocHost((int**)&A, SIZE * sizeof(float));
	cudaMallocHost((float**)&B, SIZE * sizeof(float));
	cudaMallocHost((float**)&C, SIZE * sizeof(float));
	cudaMalloc(&dev_A, SIZE * sizeof(float));
	cudaMalloc(&dev_B, SIZE * sizeof(float));
	cudaMalloc(&dev_C, SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        A[i] = (float) i;
    }
    gen_rand(B, dev_B);

    cublasSetVector(SIZE, sizeof(float), A, 1, dev_A, 1);
    cublasSetVector(SIZE, sizeof(float), B, 1, dev_B, 1);

    clock_t start = clock();
    run_cublas_calcs(A, B, C, dev_A, dev_B);

    for (int i = 0; i < SIZE; i++) {
        if (i < SIZE - 1)
            B[i] = 0;
    }

    run_cusparse_calcs(A, B, C, dev_A, dev_B);
    double pinned_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

    return pinned_time;
}

int main(int argc, char** argv) {
    double pageable_time = main_pageable();
    double pinned_time = main_pinned();

    printf("\n");
    printf("Time taken for pageable host memory: %f ms \n", pageable_time * 1000);
    printf("Time taken for pinned host memory: %f ms \n", pinned_time * 1000);
}