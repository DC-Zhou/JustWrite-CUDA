#include <stdio.h>
#include <math.h>


__global__ void add(int n, float *x, float *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
int main()
{
    int N = 1 << 20;
    float *x, *y;
    int device = -1;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaGetDevice(&device);
    // GPU prefetches unified memory memory
    cudaMemPrefetchAsync(x, N*sizeof(float), device, nullptr);
    cudaMemPrefetchAsync(y, N*sizeof(float), device, nullptr);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    // Host prefetches unified memory memory
    cudaMemPrefetchAsync(x, N*sizeof(float), cudaCpuDeviceId, nullptr);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for(int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));

    printf("Max error: %f", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}