#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

void random_init(float *data, int size)
{
    for (int i = 0; i < size; i++)
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param C          matrix C as provided to device
//! @param N          height of matrix A and matrix C
//! @param M          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix C
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////
__global__ void sgemm_gpu_kernel(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.f;
    for(int i = 0; i < K; i++)
    {
        sum += A[row * K + i] * B[i * K + col];
    }

    C[row * M + col] = alpha * sum + beta * C[row * M + col];
}

void sgemm_gpu(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);

    sgemm_gpu_kernel <<< dimGrid, dimBlock >>> (A, B, C, N, M, K, alpha, beta);
}

void performance_estimation(void(*sgemm)(const float *, const float *, float *, int, int, int, float, float), const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta) {

    int test_iterations = 100;

    StopWatchInterface *timer = 0;

    sgemm(A, B, C, N, M, K, alpha, beta);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for(int i = 0; i < test_iterations; i++) {
        sgemm(A, B, C, N, M, K, alpha, beta);
    }

    sdkStopTimer(&timer);

    float operation_time = sdkGetTimerValue(&timer);
    float operation_per_epoch = operation_time / test_iterations;

    printf("Operation time: %4f ms \n", operation_per_epoch);

    // cleanup
    sdkDeleteTimer(&timer);
}


int main()
{
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int N,M,K;
    float alpha = 2.f;
    float beta  = 1.f;

    N = M = K = 2048;

    A = (float*)malloc(N*K*sizeof(float));
    B = (float*)malloc(K*M*sizeof(float));
    C = (float*)malloc(N*M*sizeof(float));

    // allocation of gpu linear memory
    cudaMalloc((void**)&d_A, N*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*M*sizeof(float));
    cudaMalloc((void**)&d_C, N*M*sizeof(float));

    // initialization of A, B, C
    random_init(A, N*K);
    random_init(B, K*M);
    random_init(C, N*M);

    // copy A, B, C from host to device
    cudaMemcpy(d_A, A, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N*M*sizeof(float), cudaMemcpyHostToDevice);

    // do operation
    //    sgemm_gpu(d_A, d_B, d_C, N,M,K,alpha,beta);
    performance_estimation(sgemm_gpu, d_A, d_B, d_C, N,M,K,alpha,beta);

    // copy C from device to host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}