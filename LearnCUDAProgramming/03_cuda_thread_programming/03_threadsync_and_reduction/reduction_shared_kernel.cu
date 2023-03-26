#include <stdio.h>
#include "reduction.h"

//Parallel sum reduction using shared memory
//- takes log(n) steps for n input elements
//- uses n threads
//- only works for power-of-2 arrays

__global__ void reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    extern __shared__ float sdata[];

    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.0f;

    __syncthreads();

    // do reduction
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // thread synchronous reduction
        // if( (idx_x %(stride * 2)) == 0)           bandwidth: 47.50 GB/s
        if( (idx_x &(stride * 2 - 1)) == 0)          // bandwidth: 61.092 GB/s
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
        d_out[blockIdx.x] = sdata[0];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? d_in[i] : 0.0f;
    __syncthreads();
}

void reduction(float *d_out, float *d_in, int n_threads, int size)
{
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    while (size > 1)
    {
        int n_blocks = (size + n_threads - 1) / n_threads;
        reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(d_out, d_out, size);
        size = n_blocks;
    }

}