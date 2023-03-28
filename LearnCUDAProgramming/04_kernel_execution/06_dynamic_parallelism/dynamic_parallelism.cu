#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#define BLOCKDIM 256
#define BUFF_SIZE  1024

__global__ void child_kernel(int *data, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&data[idx], seed);
}

__global__ void parent_kernel(int *data)
{
    if(threadIdx.x == 0)
    {
        int child_size = BUFF_SIZE / gridDim.x;
        child_kernel<<<child_size / BLOCKDIM, BLOCKDIM>>>(&data[child_size*blockIdx.x], blockIdx.x + 1);
    }

    // Wait for all child kernels to finish
    cudaDeviceSynchronize();
}


int main()
{
    int *data;
    int num_child = 2;

    cudaMallocManaged(&data, BUFF_SIZE * sizeof(int));
    cudaMemset(data, 0, BUFF_SIZE * sizeof(int));

    parent_kernel<<<num_child, 1>>>(data);

    cudaDeviceSynchronize();

    // Count elements value
    int counter = 0;
    for (int i = 0; i < BUFF_SIZE; i++)
    {
        counter += data[i];
    }

    // getting answer
    int counter_h = 0;
    for(int i = 0; i < num_child; i++)
    {
        counter_h += (i + 1);
    }

    counter_h *= BUFF_SIZE / num_child;

    if (counter_h == counter)
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", counter, counter_h);

    cudaFree(data);

    return 0;


}