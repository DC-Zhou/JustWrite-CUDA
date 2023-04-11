#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "scan.h"
#include "utils.h"

#define SCAN_VERSION 1

void scan_host(float *h_output, float *h_input, int length, int version);

int main()
{
    srand(2022);
    float *h_input, *h_output_cpu, *h_output_gpu;
    float *d_input, *d_output;
    int length = BLOCK_DIM * 2;
}
