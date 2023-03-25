#include <stdio.h>
#include <stdlib.h>

#define N 32

void host_add(int *a, int *b, int *c){
    for(int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

// create multiple blocks
// each block will have 1 thread
// device_add<<N,1>> =>
// block0 block1
// block2 block3
__global__ void device_block_add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// create multiple threads
// device_add<<1,N>> =>
// thread0 thread1
// thread2 thread3
__global__ void device_thread_add(int *a, int *b, int *c){
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

// use in a global index
__global__ void device_global_add(int *a, int *b, int *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

// basically just fills the array with index
void fill_array(int *data) {
    for(int idx = 0; idx < N; idx++){
        data[idx] = idx;
    }
}

void print_output(int *a) {
    for (int i = 0; i < N;i++)
    printf("%d \t", a[i]);
}

void test_host_add()
{
    int *a, *b, *c;
    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    fill_array(a);
    fill_array(b);
    host_add(a, b, c);
    print_output(c);

    free(a);
    free(b);
    free(c);
}

void test_device_block_add()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    fill_array(a);
    fill_array(b);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    device_block_add<<<N, 1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_output(c);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void test_device_thread_add()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    fill_array(a);
    fill_array(b);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    device_thread_add<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_output(c);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void test_device_global_add()
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    fill_array(a);
    fill_array(b);


    cudaError_t e;
    // Copy inputs to device
    e = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    e = cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    if(e)
        printf("Error: %sn", cudaGetErrorString(e));

    // Launch add() kernel on GPU
    int threads_per_block = 8;
    int blocks_per_grid = N  / threads_per_block;
    device_global_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_output(c);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main(void){
    printf("Host array add: \n");
    test_host_add();
    printf("Device array block add  <<<N, 1>>>: \n");
    test_device_block_add();
    printf("Device array thread add <<<1, N>>>: \n");
    test_device_thread_add();
    printf("Device array add <<<4, 8>>>: \n");
    test_device_global_add();
    printf("\n");
    printf("\n");
    return 0;
}