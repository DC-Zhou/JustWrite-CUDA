#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int a[256][512][512];
__device__ float b[256][512][512];

__global__ void grid3D(int nx, int ny, int nz, int id) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int array_size = nx * ny * nz;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;
    int total_threads = block_size * grid_size;

    int thread_rank_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int thread_rank_in_grid = block_rank_in_grid * block_size + thread_rank_in_block;

    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf((float) a[z][y][x]);

    if (thread_rank_in_grid == id) {
        printf("array size %3d x %3d x %3d = %d\n",
                 nx,ny,nz, array_size);
        printf("thread block %3d x %3d x %3d = %d\n",
                  blockDim.x, blockDim.y, blockDim.z, block_size);
        printf("thread grid %3d x %3d x %3d = %d\n",
                  gridDim.x, gridDim.y, gridDim.z, grid_size);
        printf("total number of threads in grid %d\n",
                  total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",
                  z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
        printf("for thread with 3D-rank %d 1D-rank %d block rank in grid %d\n", thread_rank_in_grid, thread_rank_in_block,block_rank_in_grid);
    }


}




int main(int argc, char* argv[])
{
    int id = (argc > 1) ? atoi(argv[1]) : 12345;

    dim3 thread3D(32, 8, 2);
    dim3 block3D(16, 64, 128);

    grid3D<<<block3D, thread3D>>>(512,512,256,id);

    return 0;
}

