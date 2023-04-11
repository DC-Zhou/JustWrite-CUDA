#include "scan.h"

__global__ void scan_v2_kernel(float *d_output, float *d_input, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_data[];
    s_data[threadIdx.x] = d_input[idx];
    s_data[threadIdx.x + BLOCK_DIM] = d_input[idx + BLOCK_DIM];

    int offset = 1;

    while (offset < length) {
        __syncthreads();

        int idx_a = offset * (2 * tid + 1) - 1;
        int idx_b = offset * (2 * tid + 2) - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM) {
#if (DEBUG_INDEX > 0)
            printf("[ %d, %d]\t", idx_a, idx_b);
#endif
            s_data[idx_b] += s_data[idx_a];
        }
        offset <<= 1;
#if (DEBUG_INDEX > 0)
        if (tid == 0) printf("\n......................................\n");
#endif
    }

    offset >>= 1;
    while (offset > 0)
    {
        __syncthreads();

        int idx_a = offset * (2 * tid + 2) - 1;
        int idx_b = offset * (2 * tid + 3) - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM)
        {
#if (DEBUG_INDEX > 0)
            printf("[ %d, %d]\t", idx_a, idx_b);
#endif
            s_data[idx_b] += s_data[idx_a];
        }

        offset >>= 1;
#if (DEBUG_INDEX > 0)
        if (tid == 0) printf("\n......................................\n");
#endif
    }
    __syncthreads();

    d_output[idx] = s_data[threadIdx.x];
    d_output[idx + BLOCK_DIM] = s_data[threadIdx.x + BLOCK_DIM];
}

void scan_v2(float *d_output, float *d_input, int length)
{
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((length + BLOCK_DIM - 1) / BLOCK_DIM);

    scan_v2_kernel<<<dimGrid, dimBlock>>>(d_output, d_input, length);
}