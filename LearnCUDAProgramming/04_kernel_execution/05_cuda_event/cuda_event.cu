#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecadd_kernel(float *c, const float* a, const float *b);
void init_buffer(float *buff, int size);

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int buff_size = size * sizeof(float);

    cudaMallocHost((void**)&h_a, buff_size);
    cudaMallocHost((void**)&h_b, buff_size);
    cudaMallocHost((void**)&h_c, buff_size);

    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    cudaMalloc((void**)&d_a, buff_size);
    cudaMalloc((void**)&d_b, buff_size);
    cudaMalloc((void**)&d_c, buff_size);

    // copy host -> device
    cudaMemcpy(d_a, h_a, buff_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, buff_size, cudaMemcpyHostToDevice);

    // initialize timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // initialize cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to measure the execution time
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecadd_kernel<<< dimGrid, dimBlock >>>(d_c, d_a, d_b);

    // record the end time
    cudaEventRecord(stop, 0);

    // Synchronize the device
    cudaEventSynchronize(stop);
    sdkStopTimer(&timer);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, buff_size, cudaMemcpyDeviceToHost);

    // print estimated kernel execution time
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("CUDA event estimated - elapsed %.3f ms \n", elapsed_time_msed);

    // Compute and print the performance
    elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Host measured time= %.3f msec/s\n", elapsed_time_msed);

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;

}

void init_buffer(float *buff, int size)
{
    for (int i = 0; i < size; i++)
    {
        buff[i] = rand() / (float)RAND_MAX;
    }
}

__global__ void vecadd_kernel(float *c, const float* a, const float *b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

