//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 1
#define TOTAL_THREADS BLOCK_SIZE * NUM_BLOCKS

__device__ void copy_data_to_shared(const int * const data,
                                    int * const tmp,
                                    const int num_elements,
                                    const int tid) {
    for (int i = 0; i < num_elements; i++) {
        tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

__global__ void add(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int a_tmp = a[i];
        int b_tmp = b[i];
        int c_tmp = a_tmp + b_tmp;
        c[i] = c_tmp;
    }
}

__global__ void sub(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int a_tmp = a[i];
        int b_tmp = b[i];
        int c_tmp = a_tmp - b_tmp;
        c[i] = c_tmp;
    }
}

__global__ void mult(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int a_tmp = a[i];
        int b_tmp = b[i];
        int c_tmp = a_tmp * b_tmp;
        c[i] = c_tmp;
    }
}

__global__ void mod(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int a_tmp = a[i];
        int b_tmp = b[i];
        int c_tmp = a_tmp % b_tmp;
        c[i] = c_tmp;
    }
}

__device__ void add_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] + b[i];
    }
}

__device__ void sub_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] - b[i];
    }
}

__device__ void mult_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] * b[i];
    }
}

__device__ void mod_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] % b[i];
    }
}

__global__ void add_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[TOTAL_THREADS];
    __shared__ int b_tmp[TOTAL_THREADS];
    __shared__ int c_tmp[TOTAL_THREADS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    add_noreg(a_tmp, b_tmp, c_tmp, num_elements);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void sub_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[TOTAL_THREADS];
    __shared__ int b_tmp[TOTAL_THREADS];
    __shared__ int c_tmp[TOTAL_THREADS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    sub_noreg(a_tmp, b_tmp, c_tmp, num_elements);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void mult_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[TOTAL_THREADS];
    __shared__ int b_tmp[TOTAL_THREADS];
    __shared__ int c_tmp[TOTAL_THREADS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    mult_noreg(a_tmp, b_tmp, c_tmp, num_elements);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}

__global__ void mod_shared(int * a, int * b, int * c, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int a_tmp[TOTAL_THREADS];
    __shared__ int b_tmp[TOTAL_THREADS];
    __shared__ int c_tmp[TOTAL_THREADS];

    copy_data_to_shared(a, a_tmp, num_elements, tid);
    copy_data_to_shared(b, b_tmp, num_elements, tid);

    mod_noreg(a_tmp, b_tmp, c_tmp, num_elements);

    c[tid] = c_tmp[tid]; // copy from shared memory to global
}


// this function serves as both the encyrption and decryption function
// the negative of the encryption offset just needs to be passed in to decrpt
__global__ void caesar_cipher(char * a, char * b, int length, int offset) {
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

void run_calcs(int *a, int *b, int *c, int *dev_a, int *dev_b, int *dev_c) {
    printf("Calculations for %d block(s) of size %d threads \n", NUM_BLOCKS, 
            BLOCK_SIZE);

	printf("Addition Calculations: \n");
	add<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Subtraction Calculations: \n");
	sub<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Multiplication Calculations: \n");
	mult<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Modulo Calculations: \n");
	mod<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void shared_calcs(int *a, int *b, int *c, int *dev_a, int *dev_b, int *dev_c) {
	printf("Shared Memory Calculations for 1 block of size %d threads \n", 
            BLOCK_SIZE);
	add_shared<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
	cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Addition Calculations: \n");
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

    sub_shared<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
	cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Subtraction Calculations: \n");
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	mult_shared<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
	cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Multiplication Calculations: \n");
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	mod_shared<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
	cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Modulo Calculations: \n");
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void run_cipher(char *b, char *c, char *dev_a, char *dev_b, char *dev_c,
           int length, int offset) {
    caesar_cipher<<<NUM_BLOCKS, length>>>(dev_a, dev_b, length, offset);
    cudaMemcpy(b, dev_b, length * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Encrypted: %s \n", b);

    caesar_cipher<<<NUM_BLOCKS, length>>>(dev_b, dev_c, length, 0 - offset);
    cudaMemcpy(c, dev_c, length * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Decrypted: %s \n", c);
}

void main_pageable() {
    int *dev_a, *dev_b, *dev_c;
	int a[TOTAL_THREADS], b[TOTAL_THREADS], c[TOTAL_THREADS];
	cudaMalloc(&dev_a, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_b, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_c, TOTAL_THREADS * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < TOTAL_THREADS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);

    printf("Pageable Host Memory: \n");
    run_calcs(a, b, c, dev_a, dev_b, dev_c);

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void main_pinned() {
    int *a, *b, *c, *dev_a, *dev_b, *dev_c;
	cudaMallocHost((int**)&a, TOTAL_THREADS * sizeof(int));
	cudaMallocHost((int**)&b, TOTAL_THREADS * sizeof(int));
	cudaMallocHost((int**)&c, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_a, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_b, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_c, TOTAL_THREADS * sizeof(int));

    srand(time(NULL));
	for (int i = 0; i < TOTAL_THREADS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);

    printf("Pinned Host Memory: \n");
    run_calcs(a, b, c, dev_a, dev_b, dev_c);

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
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
    run_cipher(b, c, dev_a, dev_b, dev_c, length, 6);

	cudaFree(dev_a); 
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void main_reg_timing() {
    int *dev_a, *dev_b, *dev_c;
	int a[TOTAL_THREADS], b[TOTAL_THREADS], c[TOTAL_THREADS];
	cudaMalloc(&dev_a, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_b, TOTAL_THREADS * sizeof(int));
	cudaMalloc(&dev_c, TOTAL_THREADS * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < TOTAL_THREADS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Register Calculations: \n");
    clock_t start = clock();
    run_calcs(a, b, c, dev_a, dev_b, dev_c);
    double register_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    printf("Shared Memory Calculations: \n");
    start = clock();
    shared_calcs(a, b, c, dev_a, dev_b, dev_c);
    double shared_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	printf("Time taken with registers: \
           %f seconds. \n", register_time);
	printf("Time taken with shared memory: \
           %f seconds. \n", shared_time);

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

int main(int argc, char** argv) {
    main_pageable();
    main_pinned();
    main_caesar();
    main_reg_timing();

	return 0;
}