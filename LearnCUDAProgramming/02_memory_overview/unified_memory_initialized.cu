#include <stdio.h>
#include <math.h>

__global__ void init(int n, float *x, float *y){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

__global__ void add(int n, float *x, float *y){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        y[i] = x[i] + y[i];
    }
}



int main()
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // launch kernel on GPU on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    init<<<numBlocks, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    printf("Max error: %f", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;

}