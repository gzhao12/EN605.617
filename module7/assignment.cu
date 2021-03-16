//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 1
#define TOTAL_THREADS BLOCK_SIZE * NUM_BLOCKS

__global__ void add(int *a, int *b, int *c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sub(int *a, int *b, int *c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        c[i] = a[i] - b[i];
    }
}

__global__ void mult(int *a, int *b, int *c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        c[i] = a[i] * b[i];
    }
}

__global__ void mod(int *a, int *b, int *c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        c[i] = a[i] % b[i];
    }
}

void device_check() {
    cudaDeviceProp prop;
    int whichDevice;

    cudaGetDeviceCount(&whichDevice); 
    cudaGetDeviceProperties(&prop, whichDevice);
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

void run_calcs_streamed(int *a, int *b, int *c, int *dev_a, int *dev_b, 
                        int *dev_c, cudaStream_t stream) {
    printf("Calculations for %d block(s) of size %d threads \n", NUM_BLOCKS, 
           BLOCK_SIZE);

	printf("Addition Calculations: \n");
	add<<<NUM_BLOCKS, BLOCK_SIZE, 1, stream>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Subtraction Calculations: \n");
	sub<<<NUM_BLOCKS, BLOCK_SIZE, 1, stream>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d - %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Multiplication Calculations: \n");
	mult<<<NUM_BLOCKS, BLOCK_SIZE, 1, stream>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d * %d = %d \n", a[i], b[i], c[i]);
	}

	printf("Modulo Calculations: \n");
	mod<<<NUM_BLOCKS, BLOCK_SIZE, 1, stream>>>(dev_a, dev_b, dev_c, TOTAL_THREADS);
    cudaMemcpy(c, dev_c, TOTAL_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TOTAL_THREADS; i++) {
		printf("%d %% %d = %d \n", a[i], b[i], c[i]);
	}
}

double main_pageable() {
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

    clock_t start = clock();
	cudaMemcpy(dev_a, a, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice);

    printf("Pageable Host Memory: \n");
    run_calcs(a, b, c, dev_a, dev_b, dev_c);
    double pageable_time = ((double) (clock() - start)) / CLOCKS_PER_SEC;

    cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

    return pageable_time;
}

float main_streamed() {
    int *a, *b, *c, *dev_a, *dev_b, *dev_c;
    cudaEvent_t start, stop; 
    float elapsedTime; 

    device_check();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream; 
    cudaStreamCreate(&stream); 

    cudaMalloc((void**)&dev_a, TOTAL_THREADS * sizeof(*dev_a)); 
    cudaMalloc((void**)&dev_b, TOTAL_THREADS * sizeof(*dev_b)); 
    cudaMalloc((void**)&dev_c, TOTAL_THREADS * sizeof(*dev_c)); 

    cudaHostAlloc((void**)&a, TOTAL_THREADS * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, TOTAL_THREADS * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, TOTAL_THREADS * sizeof(int), cudaHostAllocDefault);

    srand(time(NULL));
	for (int i = 0; i < TOTAL_THREADS; i++) {
		a[i] = i;
		b[i] = rand() % 3;
	}

    cudaEventRecord(start);

    cudaMemcpyAsync(dev_a, a, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(dev_b, b, TOTAL_THREADS * sizeof(int), cudaMemcpyHostToDevice, stream); 

    printf("Streamed host memory calculations: \n");
    run_calcs_streamed(a, b, c, dev_a, dev_b, dev_c, stream);
  
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    cudaFreeHost(a); 
    cudaFreeHost(b); 
    cudaFreeHost(c); 
    cudaFree(dev_a); 
    cudaFree(dev_b); 
    cudaFree(dev_c);

    return elapsedTime;
}

int main(int argc, char** argv)
{
    double pageable_time = main_pageable();
    float streamed_time = main_streamed();

    printf("Time taken for streaming from host: %f ms \n", pageable_time * 1000);
    printf("Time taken for pageable host memory: %f ms \n", streamed_time);
}
