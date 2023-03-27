#include <cstdio>
#include <cuda_runtime_api.h>
#include <helper_timer.h>

using namespace std;

void init_buffer(float *buff, int size)
{
    for (int i = 0; i < size; i++) {
        buff[i] = rand() / (float)RAND_MAX;
    }
}

__global__ void vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

class Operator
{
private:
    int _index;
    cudaStream_t stream;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *data);

    void print_time();

public:
    Operator() {
        cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);
    }

    ~Operator() {
        cudaStreamDestroy(stream);
        sdkDeleteTimer(&p_timer);
    }

    void set_index(int index) {
        _index = index;
    }

    void async_operation(float *h_c, const float *h_a, const float *h_b,
                         float *d_c, float *d_a, float *d_b,
                         const int size, const int buff_size);

};

void Operator::Callback(cudaStream_t stream, cudaError_t status, void *data) {
    Operator *op = (Operator*)data;
    op->print_time();
}

void Operator::print_time() {
    sdkStopTimer(&p_timer);    // end timer
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", _index, elapsed_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                               float *d_c, float *d_a, float *d_b,
                               const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int buff_size = size * sizeof(float);
    int n_operator = 4;

    // initialize timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // allocate host memory
    cudaMallocHost((void**)&h_a, buff_size);
    cudaMallocHost((void**)&h_b, buff_size);
    cudaMallocHost((void**)&h_c, buff_size);

    // initialize host memory
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memory
    cudaMalloc((void**)&d_a, buff_size);
    cudaMalloc((void**)&d_b, buff_size);
    cudaMalloc((void**)&d_c, buff_size);

    // create list of operation elements
    Operator *ls_operator = new Operator[n_operator];


    sdkStartTimer(&timer);

    // execute each operator collesponding data
    for (int i = 0; i < n_operator; i++) {
        int offset = i * size / n_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / n_operator, buff_size / n_operator);
    }

    // synchronize all the stream operation
    cudaDeviceSynchronize();

    sdkStopTimer(&timer);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * buff_size * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    // delete timer
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