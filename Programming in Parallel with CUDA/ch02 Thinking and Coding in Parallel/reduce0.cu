#include "common.h"

__global__ void reduce0(float *x, int m)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}


int main(int argc, char* argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 20;

    thrust::host_vector<float>     h_vec(N);
    thrust::device_vector<float>   d_vec(N);

    // random initialize input
    std::default_random_engine gen(2022);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int i = 0; i < N; i++) h_vec[i] = fran(gen);
    // H2D copy
    d_vec = h_vec;

    // thrust::reduce
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    double thrust_reduce_sum = thrust::reduce(h_vec.begin(), h_vec.end());
    sdkStopTimer(&timer);
    printf("thrust reduce time: %f ms\n", sdkGetTimerValue(&timer));
    sdkResetTimer(&timer);

    // cpu::reduce
    double cpu_reduce_sum = 0.0;
    for(int k = 0; k < N; k++) cpu_reduce_sum += h_vec[k];

    // gpu::reduce
    sdkStartTimer(&timer);
    for(int m = N/2; m > 0; m /= 2)
    {
        int threads = std::min(256, m);
        int blocks =  std::max(m / 256, 1);
        reduce0<<<blocks, threads>>>(d_vec.data().get(), m);
    }
    sdkStopTimer(&timer);
    printf("gpu reduce time: %f ms\n", sdkGetTimerValue(&timer));
    sdkResetTimer(&timer);
    sdkDeleteTimer(&timer);

    double gpu_reduce_sum = d_vec[0];
    sdkStopTimer(&timer);


    printf("thrust reduce sum: %f,\n"
                  "cpu reduce    sum: %f,\n"
                  "gpu reduce    sum: %f,\n",thrust_reduce_sum, cpu_reduce_sum, gpu_reduce_sum);
    // time
}