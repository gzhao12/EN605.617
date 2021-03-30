#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <iostream>
#include <chrono>

#define MAX 25
#define SIZE 512

template<typename T>
struct rand_btwn {
    __host__ T operator()(T& VecElem) const { return (T)rand() % MAX; }
};

void run_calcs(thrust::host_vector<int> a, thrust::host_vector<int> b, 
               thrust::device_vector<int> da, thrust::device_vector<int> db,
               thrust::device_vector<int> dc) {
    // addition
    thrust::transform(da.begin(), da.end(), db.begin(), dc.begin(),
                      thrust::plus<int>());
	for (int i = 0; i < SIZE; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << dc[i] << std::endl;
	}
    // subtraction
    thrust::transform(da.begin(), da.end(), db.begin(), dc.begin(),
                      thrust::minus<int>());
	for (int i = 0; i < SIZE; i++) {
        std::cout << a[i] << " - " << b[i] << " = " << dc[i] << std::endl;
	}
    // multiplication
    thrust::transform(da.begin(), da.end(), db.begin(), dc.begin(),
                      thrust::multiplies<int>());
	for (int i = 0; i < SIZE; i++) {
        std::cout << a[i] << " * " << b[i] << " = " << dc[i] << std::endl;
	}
    // modulus
    thrust::transform(da.begin(), da.end(), db.begin(), dc.begin(),
                      thrust::modulus<int>());
	for (int i = 0; i < SIZE; i++) {
        std::cout << a[i] << " % " << b[i] << " = " << dc[i] << std::endl;
	}
}

double main_pageable() {
    // initialize host vectors
    thrust::host_vector<int> a(SIZE);
    thrust::host_vector<int> b(SIZE);

    // make an index vector and a randomly generated vector
    thrust::counting_iterator<int> iter(0);
    thrust::copy(iter, iter + a.size(), a.begin());
    thrust::transform(thrust::host, b.begin(), b.end(), b.begin(), rand_btwn<int>());

    // copy to device
    thrust::device_vector<int> da = a;
    thrust::device_vector<int> db = b;
    thrust::device_vector<int> dc(SIZE);

    std::cout << "Pageable host memory calculations:" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_calcs(a, b, da, db, dc);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    return duration.count();
}

double main_pinned() {
    // initialize host vectors
    std::vector<int, thrust::system::cuda::experimental::pinned_allocator<int>> a(SIZE);
    std::vector<int, thrust::system::cuda::experimental::pinned_allocator<int>> b(SIZE);

    // make an index vector and a randomly generated vector
    thrust::counting_iterator<int> iter(0);
    thrust::copy(iter, iter + a.size(), a.begin());
    thrust::transform(thrust::host, b.begin(), b.end(), b.begin(), rand_btwn<int>());

    // copy to device
    thrust::device_vector<int> da = a;
    thrust::device_vector<int> db = b;
    thrust::device_vector<int> dc(SIZE);

    std::cout << "Pinned host memory calculations:" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_calcs(a, b, da, db, dc);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    return duration.count();
}

int main() {
    auto pageable_time = main_pageable();
    auto pinned_time = main_pinned();
    
    std::cout << "Time taken for pageable host memory: " << pageable_time << " ms" << std::endl;
    std::cout << "Time taken for pinned host memory: " << pinned_time << " ms" << std::endl;
}