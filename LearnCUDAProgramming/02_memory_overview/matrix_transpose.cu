#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define BLOCK_SIZE 32

void fill_array(int *data) {
    for(int idx=0;idx<(N*N);idx++)
        data[idx] = idx;
}

__global__ void transpose(int *input, int *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = x + y * N;
    int transposeIndex = x * N + y;
    output[transposeIndex] = input[index];
}

__global__ void transpose_shared(int *input, int *output) {
    __shared__ int sharedMemory [BLOCK_SIZE][BLOCK_SIZE];

    // global index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // transposed global memory index
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    int ty = blockIdx.x * blockDim.y + threadIdx.y;

    // local index
    int localx = threadIdx.x;
    int localy = threadIdx.y;

    int index = x + y * N;
    int transposeIndex = ty * N + tx;

    // reading from global memory in coalesed manner and performing tanspose in shared memory
    sharedMemory[localx][localy] = input[index];

    __syncthreads();

    // writing into global memory in coalesed fashion via transposed data in shared memory
    output[transposeIndex] = sharedMemory[localy][localx];
}

void print_output(int *a, int *b) {
    printf("\n Original Matrix::\n");
    for(int idx=0;idx<(N*N);idx++) {
        if(idx%N == 0)
            printf("\n");
        printf(" %d ",  a[idx]);
    }
    printf("\n Transposed Matrix::\n");
    for(int idx=0;idx<(N*N);idx++) {
        if(idx%N == 0)
            printf("\n");
        printf(" %d ",  b[idx]);
    }
}

bool check_equal(const int *a, const int *b, int size)
{
    for(int idx=0;idx<size;idx++)
        if(a[idx] != b[idx])
            return false;
    return true;
}

int main(){
    int *a, *naive_b, *shared_b;
    int *d_a, *d_b;

    int size = N * N * sizeof(int);

    a = (int *)malloc(size);
    fill_array(a);

    naive_b = (int *)malloc(size);
    shared_b = (int *)malloc(size);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, naive_b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize(N/BLOCK_SIZE, N/BLOCK_SIZE, 1);

    transpose<<<gridSize, blockSize>>>(d_a, d_b);

    cudaMemcpy(naive_b, d_b, size, cudaMemcpyDeviceToHost);

    cudaFree(d_b);
    cudaMalloc((void **)&d_b, size);

    transpose_shared<<<gridSize, blockSize>>>(d_a, d_b);

    cudaMemcpy(shared_b, d_b, size, cudaMemcpyDeviceToHost);

    if(check_equal(naive_b, shared_b, N*N))
        printf("naive and shared memory transpose are equal !");

    cudaFree(d_a);
    cudaFree(d_b);

    free(a);
    free(naive_b);
    free(shared_b);
    return 0;
}

