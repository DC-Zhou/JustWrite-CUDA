#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu() {
    printf("Hello, World from thread [%d, %d]! \t From device \n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello, World from host!\n");
    print_from_gpu<<<1, 1>>>();
    // [0,0]
    print_from_gpu<<<2, 1>>>();
    // [0,0] - [0,1]
    print_from_gpu<<<1, 2>>>();
    // [0,0] - [1,0]
    cudaDeviceSynchronize();
    return 0;
}
