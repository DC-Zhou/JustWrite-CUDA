//
// Created by Zhou on 2023/3/31.
//

#ifndef CH01_CUDA_SUM_H
#define CH01_CUDA_SUM_H

#include <cstdio>
#include <cstdlib>

void gpu_sin(float *sums, int steps, int terms, float step_size);

inline float sin_sum(float x, int term);

#endif //CH01_CUDA_SUM_H
