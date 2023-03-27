#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float *a, const float *b);
void init_buffer(float *buf, int n);

class Operator
{
private:
    int index;
    cudaStream_t stream;

public:
    Operator(){
        cudaStreamCreate(&stream);
    }

    ~Operator(){
        cudaStreamDestroy(stream);
    }

    void set_index(int idx){
        index = idx;
    }

    void async_operation(float *h_c, const float *h_a, const float *h_b,
                         float *d_c, float *d_a, float *d_b,
                         const int size, const int bufsize);

};  // Operator


void Operator::async_operation(float *h_c, const float *h_a, const float *h_b, float *d_c, float *d_a, float *d_b,
                               const int size, const int bufsize) {
    // copy data from host to device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch kernel
    int block_size = 256;
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(size / dimBlock.x, 1, 1);

    vecAdd_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_c, d_a, d_b);

    // copy data from device to host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // synchronize
    cudaStreamSynchronize(stream);

    printf("Launched GPU task %d\n", index);

}

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);

    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);

    cudaMallocHost((void **)&h_a, bufsize);
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    cudaMalloc((void **)&d_a, bufsize);
    cudaMalloc((void **)&d_b, bufsize);
    cudaMalloc((void **)&d_c, bufsize);

    // create list of operation elements
    Operator *ls_operator = new Operator[num_operator];

    // create timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    cudaDeviceSynchronize();

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;

}

void init_buffer(float *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX;
}

__global__ void vecAdd_kernel(float *c, const float *a, const float *b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    c[idx] = a[idx] + b[idx];
}
