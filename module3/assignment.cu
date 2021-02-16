//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

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

void add_host(int * a, int * b, int * c, int totalThreads) {
	std::cout << totalThreads << std::endl;
	for (int i = 0; i < totalThreads; i++) { 
		c[i] = a[i] + b[i];
	}
}

void main_add(int totalThreads, int blockSize, int numBlocks) {
	int a[totalThreads], b[totalThreads], c[totalThreads];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_b, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	add<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < totalThreads; i++) {
		std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
	}
}

void main_sub(int totalThreads, int blockSize, int numBlocks) {
	int a[totalThreads], b[totalThreads], c[totalThreads];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_b, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	subtract<<<numBlocks, blockSize>>>(dev_b, dev_a, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < totalThreads; i++) {
		std::cout << a[i] << " - " << b[i] << " = " << c[i] << std::endl;
	}
}

void main_mult(int totalThreads, int blockSize, int numBlocks) {
	int a[totalThreads], b[totalThreads], c[totalThreads];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_b, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	mult<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < totalThreads; i++) {
		std::cout << a[i] << " * " << b[i] << " = " << c[i] << std::endl;
	}
}

void main_mod(int totalThreads, int blockSize, int numBlocks) {
	int a[totalThreads], b[totalThreads], c[totalThreads];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_b, totalThreads * sizeof(int));
	cudaMalloc((void**)&dev_c, totalThreads * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

	cudaMemcpy(dev_a, a, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	mod<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < totalThreads; i++) {
		std::cout << a[i] << " % " << b[i] << " = " << c[i] << std::endl;
	}
}

int main(int argc, char** argv)
{
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

	std::cout << "Calculations for " << numBlocks << " block(s) of size "
		<< blockSize << " threads." << std::endl;

	std::cout << "Addition Calculations:" << std::endl;
	main_add(totalThreads, blockSize, numBlocks);

	std::cout << "Subtraction Calculations:" << std::endl;
	main_sub(totalThreads, blockSize, numBlocks);

	std::cout << "Multiplication Calculations:" << std::endl;
	main_mult(totalThreads, blockSize, numBlocks);

	std::cout << "Modulo Calculations:" << std::endl;
	main_mod(totalThreads, blockSize, numBlocks);

	return 0;
}