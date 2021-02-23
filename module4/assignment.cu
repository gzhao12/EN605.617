//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void subtract(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = b[i] - a[i];
}

__global__ void mult(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void mod(int * a, int * b, int * c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] % b[i];
}

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

void main_add(int totalThreads, int blockSize, int numBlocks, int *a, int *b,
			  int *c, int *dev_a, int *dev_b, int *dev_c) {
	add<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < totalThreads; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}
}

void main_sub(int totalThreads, int blockSize, int numBlocks, int *a, int *b,
			  int *c, int *dev_a, int *dev_b, int *dev_c) {
	subtract<<<numBlocks, blockSize>>>(dev_b, dev_a, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < totalThreads; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}
}

void main_mult(int totalThreads, int blockSize, int numBlocks, int *a, int *b,
			  int *c, int *dev_a, int *dev_b, int *dev_c) {
	mult<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < totalThreads; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}
}

void main_mod(int totalThreads, int blockSize, int numBlocks, int *a, int *b,
			  int *c, int *dev_a, int *dev_b, int *dev_c) {
	mod<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < totalThreads; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

void main_pageable(int totalThreads, int blockSize, int numBlocks) {
	int *a, *b, *c, *dev_a, *dev_b, *dev_c;
	a = (int*)malloc(totalThreads * sizeof(int));
	b = (int*)malloc(totalThreads * sizeof(int));
	c = (int*)malloc(totalThreads * sizeof(int));
	cudaMalloc(&dev_a, totalThreads * sizeof(int));
	cudaMalloc(&dev_b, totalThreads * sizeof(int));
	cudaMalloc(&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	printf("Calculations for %d block(s) of size %d threads \n", numBlocks, blockSize);
	printf("Addition Calculations: \n");
	main_add(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Subtraction Calculations: \n");
	main_sub(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Multiplication Calculations: \n");
	main_mult(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Modulo Calculations: \n");
	main_mod(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	free(a);
	free(b);
	free(c);
}

void main_pinned(int totalThreads, int blockSize, int numBlocks) {
	int *a, *b, *c, *dev_a, *dev_b, *dev_c;
	cudaMallocHost((int**)&a, totalThreads*sizeof(int));
	cudaMallocHost((int**)&b, totalThreads*sizeof(int));
	cudaMallocHost((int**)&c, totalThreads*sizeof(int));
	cudaMalloc(&dev_a, totalThreads * sizeof(int));
	cudaMalloc(&dev_b, totalThreads * sizeof(int));
	cudaMalloc(&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	printf("Calculations for %d block(s) of size %d threads \n", numBlocks, blockSize);
	printf("Addition Calculations: \n");
	main_add(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Subtraction Calculations: \n");
	main_sub(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Multiplication Calculations: \n");
	main_mult(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);
	printf("Modulo Calculations: \n");
	main_mod(totalThreads, blockSize, numBlocks, a, b, c, dev_a, dev_b, dev_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

void host_caesar_cipher(char *a, char *b, int length, int offset) {
	char *dev_a, *dev_b;
	cudaMalloc(&dev_a, length * sizeof(char));
	cudaMalloc(&dev_b, length * sizeof(char));

	cudaMemcpy(dev_a, a, length * sizeof(char), cudaMemcpyHostToDevice);

	// always one block with length number of threads
	caesar_cipher<<<length, length>>>(dev_a, dev_b, length, offset);
	cudaMemcpy(b, dev_b, length * sizeof(char), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
}

void main_caesar() {
	char a []= "Hello World";
	int length = sizeof(a)/sizeof(a[0]);
	char *b, *c, *a_pinned, *b_pinned, *c_pinned;
	
	b = (char*)malloc(length * sizeof(char));
	c = (char*)malloc(length * sizeof(char));
	cudaMallocHost((char**)&a_pinned, length * sizeof(char));
	cudaMallocHost((char**)&b_pinned, length * sizeof(char));
	cudaMallocHost((char**)&c_pinned, length * sizeof(char));
	memcpy(a_pinned, a, length * sizeof(char));

	clock_t start = clock();
	host_caesar_cipher(a, b, length, 6);
	host_caesar_cipher(b, c, length, -6);
	double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;
	printf("Pageable, encrypted: %s \n", b);
	printf("Pageable, decrypted: %s \n", c);

	start = clock();
	host_caesar_cipher(a_pinned, b_pinned, length, 6);
	host_caesar_cipher(b_pinned, c_pinned, length, -6);
	double pinned_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;
	printf("Pinned, encrypted: %s \n", b_pinned);
	printf("Pinned, decrypted: %s \n", c_pinned);

	printf("Time taken for pageable memory: %f seconds. \n", pageable_time);
	printf("Time taken for pinned memory: %f seconds. \n", pinned_time);

	cudaFreeHost(a_pinned);
	cudaFreeHost(b_pinned);
	cudaFreeHost(c_pinned);
	free(b);
	free(c);
}

int main(int argc, char** argv) {
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	clock_t start = clock();
	main_pageable(totalThreads, blockSize, numBlocks);
	double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;
	start = clock();
	main_pinned(totalThreads, blockSize, numBlocks);
	double pinned_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

	printf("Time taken for pageable memory: %f seconds. \n", pageable_time);
	printf("Time taken for pinned memory: %f seconds. \n", pinned_time);

	main_caesar();

	return 0;
}