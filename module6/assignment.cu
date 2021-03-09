//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ELEMENTS 256
#define NUM_BLOCKS 2

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

__global__ void add_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sub_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] - b[i];
    }
}

__global__ void mult_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] * b[i];
    }
}

__global__ void mod_noreg(int * a, int * b, int * c, int num_elements) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
        c[i] = a[i] % b[i];
    }
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
            NUM_ELEMENTS);

	printf("Addition Calculations: \n");
	add<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Subtraction Calculations: \n");
	sub<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Multiplication Calculations: \n");
	mult<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Modulo Calculations: \n");
	mod<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void run_noreg_calcs(int *a, int *b, int *c, int *dev_a, int *dev_b, int *dev_c) {
    printf("Calculations for %d block(s) of size %d threads \n", NUM_BLOCKS, 
            NUM_ELEMENTS);

	printf("Addition Calculations: \n");
	add<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Subtraction Calculations: \n");
	sub<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Multiplication Calculations: \n");
	mult<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Modulo Calculations: \n");
	mod<<<NUM_BLOCKS, NUM_ELEMENTS>>>(dev_a, dev_b, dev_c, NUM_ELEMENTS);
    cudaMemcpy(c, dev_c, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < NUM_ELEMENTS; i++) {
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

    run_calcs(a, b, c, dev_a, dev_b, dev_c);

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

    clock_t start = clock();
    run_calcs(a, b, c, dev_a, dev_b, dev_c);
    double register_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    start = clock();
    run_noreg_calcs(a, b, c, dev_a, dev_b, dev_c);
    double noreg_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	printf("Time taken with registers: \
           %f seconds. \n", register_time);
	printf("Time taken without registers: \
           %f seconds. \n", noreg_time);

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

int main(int argc, char** argv) {
    clock_t start = clock();
    main_pageable();
    double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    start = clock();
    main_pinned();
	double pinned_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	printf("Time taken for pageable memory: \
           %f seconds. \n", pageable_time);
	printf("Time taken for pinned memory: \
           %f seconds. \n", pinned_time);

    main_caesar();

    // main_reg_timing();

	return 0;
}