//
// Created by Zhou on 2023/3/26.
//

#ifndef WARP_SYNCHRONOUS_PROGRAMMING_REDUCTION_H
#define WARP_SYNCHRONOUS_PROGRAMMING_REDUCTION_H

// @reduction_kernel.cu
void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#endif //WARP_SYNCHRONOUS_PROGRAMMING_REDUCTION_H
