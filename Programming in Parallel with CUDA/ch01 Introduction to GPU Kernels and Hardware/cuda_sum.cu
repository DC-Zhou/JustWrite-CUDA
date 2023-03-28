
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "cuda_runtime.h"
#include "thrust/device_vector.h"

__host__ __device__ inline float sin_sum(float x, int terms);
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size);


int main(int argc, char* argv[])
{
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;

    double pi = 3.141592653;
    double step_size = pi / (steps - 1);

    // allocate memory on the device
    thrust::device_vector<float> d_sums(steps);
    float *d_sums_ptr = thrust::raw_pointer_cast(d_sums.data());

    // starct timing
    auto start = std::chrono::steady_clock::now();
    gpu_sin<<<blocks, threads>>>(d_sums_ptr, steps, terms, (float)step_size);

    double gpu_sum = thrust::reduce(d_sums.begin(), d_sums.end());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Trapezoidal Rule correction
    gpu_sum -= 0.5*(sin_sum(0.0,terms)+sin_sum(pi, terms));
    gpu_sum *= step_size;
    printf("gpu sum = %.10f,steps %d terms %d time %.3f ms\n",
           gpu_sum, steps, terms, diff);

    return 0;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size)
{
    // unique thread index
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if(step < steps) {
        float x = step * step_size;
        sums[step] = sin_sum(x, terms);
    }
}

__host__ __device__ inline float sin_sum(float x, int terms)
{
    float x2 = x*x;
    float term = x;
    float sum = term;

    for(int n = 1; n < terms; n++) {
        term *= -x2 / ((2*n)*(2*n+1));
        sum += term;
    }
    return sum;
}
