#include <cstdio>

using namespace std;

__global__ void fool_kernel(int step) {
    printf("Loop: %d \n", step);
}

int main()
{
    int n_steps = 10;

    for (int i = 0; i < n_steps; i++) {
        fool_kernel<<<1, 1, 0, 0>>>(i);
    }

    cudaDeviceSynchronize();

    return 0;
}