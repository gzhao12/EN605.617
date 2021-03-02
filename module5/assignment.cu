//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ELEMENTS 128
__constant__ int a_const[NUM_ELEMENTS];
__constant__ int b_const[NUM_ELEMENTS];
__constant__ int c_const[NUM_ELEMENTS];
__constant__ char caesar_0[NUM_ELEMENTS];
__constant__ char caesar_1[NUM_ELEMENTS];
__constant__ char caesar_2[NUM_ELEMENTS];

__device__ void copy_data_to_shared(const int * const data,
                                    int * const tmp,
                                    const int num_elements,
                                    const int tid) {
    for (int i = 0; i < num_elements; i++) {
        tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

__device__ void copy_char_to_shared(const char * const data,
                                    char * const tmp,
                                    const int num_elements,
                                    const int tid) {
    for (int i = 0; i < num_elements; i++) {
        tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

__device__ void add_calc(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ void sub_calc(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] - b[i];
}

__device__ void mult_calc(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
}

__device__ void mod_calc(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] % b[i];
}

__device__ void caesar_cipher(char * a, char * b, int length, int offset) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		char tmp = a[i];
		if (offset > 0) {
			if (tmp >= 'a' && tmp <= 'z') {
				tmp = tmp + offset;
				if (tmp > 'z') {
					tmp = tmp - 'z' + 'a' - 1;
				}
			}
			else if (tmp >= 'A' && tmp <= 'Z') {
				tmp = tmp + offset;
				if (tmp > 'Z') {
					tmp = tmp - 'Z' + 'A' - 1;
				}
			}
			b[i] = tmp;
		}
		else {
			if (tmp >= 'a' && tmp <= 'z') {
				tmp = tmp + offset;
				if (tmp < 'A') {
					tmp = tmp + 'z' - 'a' + 1;
				}
			}
			else if (tmp >= 'A' && tmp <= 'Z') {
				tmp = tmp + offset;
				if (tmp < 'A') {
					tmp = tmp + 'Z' - 'A' + 1;
				}
			}
			b[i] = tmp;
		}
	}
}

__global__ void add_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[NUM_ELEMENTS];
    __shared__ int b_tmp[NUM_ELEMENTS];
    __shared__ int c_tmp[NUM_ELEMENTS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    add_calc(a_tmp, b_tmp, c_tmp);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void sub_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[NUM_ELEMENTS];
    __shared__ int b_tmp[NUM_ELEMENTS];
    __shared__ int c_tmp[NUM_ELEMENTS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    sub_calc(a_tmp, b_tmp, c_tmp);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void mult_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[NUM_ELEMENTS];
    __shared__ int b_tmp[NUM_ELEMENTS];
    __shared__ int c_tmp[NUM_ELEMENTS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    mult_calc(a_tmp, b_tmp, c_tmp);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void mod_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[NUM_ELEMENTS];
    __shared__ int b_tmp[NUM_ELEMENTS];
    __shared__ int c_tmp[NUM_ELEMENTS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    mod_calc(a_tmp, b_tmp, c_tmp);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void add_const(int * dev_c) {
    add_calc(a_const, b_const, dev_c);
}

__global__ void sub_const(int * dev_c) {
    sub_calc(a_const, b_const, dev_c);
}

__global__ void mult_const(int * dev_c) {
    mult_calc(a_const, b_const, dev_c);
}

__global__ void mod_const(int * dev_c) {
    mod_calc(a_const, b_const, dev_c);
}

__global__ void caesar_shared(char * a, char * b, int length, int offset) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ char a_tmp[NUM_ELEMENTS];
    __shared__ char b_tmp[NUM_ELEMENTS];

    copy_char_to_shared(a, a_tmp, length, tid);

    caesar_cipher(a_tmp, b_tmp, length, offset);

    b[tid] = b_tmp[tid];
}

__global__ void caesar_const(char * a, char * b, int length, int offset) {
    caesar_cipher(a, b, length, offset);
}

void host_shared_main(int *a, int *b, int *c, int *dev_a, int *dev_b, int *dev_c) {
	printf("Shared Memory Calculations for 1 block of size %d threads \n", 
            NUM_ELEMENTS);
	add_shared<<<1, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
	cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Addition Calculations: \n");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

    sub_shared<<<1, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
	cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Subtraction Calculations: \n");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	mult_shared<<<1, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
	cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Multiplication Calculations: \n");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	mod_shared<<<1, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
	cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Modulo Calculations: \n");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void host_const_main(int *a, int *b, int *c, int *dev_c) {
    printf("Constant Memory Calculations for 1 block of size %d threads \n", 
            NUM_ELEMENTS);
	printf("Addition Calculations: \n");
	add_const<<<1, NUM_ELEMENTS>>>(dev_c);
    cudaMemcpyToSymbol(c_const, dev_c, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyFromSymbol(c, c_const, NUM_ELEMENTS * sizeof(int));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}
	printf("Subtraction Calculations: \n");
	sub_const<<<1, NUM_ELEMENTS>>>(dev_c);
    cudaMemcpyToSymbol(c_const, dev_c, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyFromSymbol(c, c_const, NUM_ELEMENTS * sizeof(int));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}
	printf("Multiplication Calculations: \n");
	mult_const<<<1, NUM_ELEMENTS>>>(dev_c);
    cudaMemcpyToSymbol(c_const, dev_c, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyFromSymbol(c, c_const, NUM_ELEMENTS * sizeof(int));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}
	printf("Modulo Calculations: \n");
	mod_const<<<1, NUM_ELEMENTS>>>(dev_c);
    cudaMemcpyToSymbol(c_const, dev_c, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyFromSymbol(c, c_const, NUM_ELEMENTS * sizeof(int));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void host_caesar_shared(char *b, char *c, char *dev_a, char *dev_b, char *dev_c, 
                        int length, int offset) {
    caesar_shared<<<1, length>>>(dev_a, dev_b, length, offset);
    cudaMemcpy(b, dev_b, length * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Shared, encrypted: %s \n", b);

    caesar_shared<<<1, length>>>(dev_b, dev_c, length, 0 - offset);
    cudaMemcpy(c, dev_c, length * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Shared, decrypted: %s \n", c);
}

void host_caesar_const(char *a, char *b, char *c, int length, int offset) {
    cudaMemcpyToSymbol(caesar_0, a, length * sizeof(char));

    caesar_const<<<1, length>>>(caesar_0, b, length, offset);
    cudaMemcpyToSymbol(caesar_1, b, length * sizeof(char));
    cudaMemcpyFromSymbol(b, caesar_1, length * sizeof(char));

    caesar_const<<<1, length>>>(caesar_1, c, length, 0 - offset);
    cudaMemcpyToSymbol(caesar_2, c, length * sizeof(char));
    cudaMemcpyFromSymbol(c, caesar_2, length * sizeof(char));

    printf("Constant, encrypted: %s \n", b);
    printf("Constant, decrypted: %s \n", c);
}

void main_caesar() {
    char a [] = "Hello World";
	int length = sizeof(a)/sizeof(a[0]);
    char b[length], c[length];
    char *dev_a, *dev_b, *dev_c;

    cudaMalloc(&dev_a, length * sizeof(char));
    cudaMalloc(&dev_b, length * sizeof(char));
    cudaMalloc(&dev_c, length * sizeof(char));

    cudaMemcpy(dev_a, a, length * sizeof(char), cudaMemcpyHostToDevice);

    host_caesar_shared(b, c, dev_a, dev_b, dev_c, length, 6);
    host_caesar_const(a, b, c, length, 6);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void main_pageable() {
	int *dev_a, *dev_b, *dev_c;
	int a[NUM_ELEMENTS], b[NUM_ELEMENTS], c[NUM_ELEMENTS];
	cudaMalloc(&dev_a, NUM_ELEMENTS * sizeof(int));
	cudaMalloc(&dev_b, NUM_ELEMENTS * sizeof(int));
	cudaMalloc(&dev_c, NUM_ELEMENTS * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_const, a, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyToSymbol(b_const, b, NUM_ELEMENTS * sizeof(int));

    host_shared_main(a, b, c, dev_a, dev_b, dev_c);
    host_const_main(a, b, c, dev_c);

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void main_pinned() {
    int *a, *b, *c, *dev_a, *dev_b, *dev_c;
	cudaMallocHost((int**)&a, NUM_ELEMENTS * sizeof(int));
	cudaMallocHost((int**)&b, NUM_ELEMENTS * sizeof(int));
	cudaMallocHost((int**)&c, NUM_ELEMENTS * sizeof(int));
	cudaMalloc(&dev_a, NUM_ELEMENTS * sizeof(int));
	cudaMalloc(&dev_b, NUM_ELEMENTS * sizeof(int));
	cudaMalloc(&dev_c, NUM_ELEMENTS * sizeof(int));

    srand(time(NULL));
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_const, a, NUM_ELEMENTS * sizeof(int));
    cudaMemcpyToSymbol(b_const, b, NUM_ELEMENTS * sizeof(int));

    host_shared_main(a, b, c, dev_a, dev_b, dev_c);
    host_const_main(a, b, c, dev_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

int main(int argc, char** argv) {
	clock_t start = clock();
    main_pageable();
	double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	start = clock();
    main_pinned();
	double pinned_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	printf("Time taken for pageable memory: %f seconds. \n", pageable_time);
	printf("Time taken for pinned memory: %f seconds. \n", pinned_time);

    main_caesar();

	return 0;
}