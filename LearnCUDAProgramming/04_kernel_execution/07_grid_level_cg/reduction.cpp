//
// Created by Zhou on 2023/3/28.
//

#include "reduction.h"

#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>
#include <helper_timer.h>

void run_benchmark(int (*reduce)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size);

void init_input(float *data, int size);

float get_cpu_result(float *data, int size);

// cooperative groups support
int check_cooperative_launch_support()
{
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    if(deviceProp.cooperativeLaunch == 0)
        return 0;
    return 1;
}

int main()
{

    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;

    float result_host, result_device;

    // check device availability
    if(check_cooperative_launch_support() == 0)
    {
        puts("Target GPU does not support Cooperative Kernel Launch. Exit.");
        exit(EXIT_FAILURE);
    }

    srand(2019);

    h_inPtr = (float *)malloc(size * sizeof(float));

    init_input(h_inPtr, size);

    cudaMalloc((void **)&d_inPtr, size * sizeof(float));
    cudaMalloc((void **)&d_outPtr, sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

    run_benchmark(reduction_grid_sync, d_outPtr, d_inPtr, size);
    cudaMemcpy(&result_device, d_outPtr, sizeof(float), cudaMemcpyDeviceToHost);

    // get all sum from memoru
    result_host = get_cpu_result(h_inPtr, size);
    printf("Host result: %f, Device result: %f\n", result_host, result_device);

    cudaFree(d_inPtr);
    cudaFree(d_outPtr);
    free(h_inPtr);

    return 0;
}

void init_input(float *data, int size)
{
    for(int i = 0; i < size; i++)
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

float get_cpu_result(float *data, int size)
{
    float sum = 0.f;
    for(int i = 0; i < size; i++)
        sum += data[i];
    return (float)sum;
}

void run_benchmark(int (*reduce)(float*, float*, int, int),
                   float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    // warm-up
    reduce(d_outPtr, d_inPtr, size, num_threads);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for(int i = 0; i < test_iter; i++)
        reduce(d_outPtr, d_inPtr, size, num_threads);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);
}

