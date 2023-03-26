//
// Created by Zhou on 2023/3/26.
//

#ifndef INC_10_ATOMIC_OPERATION_REDUCTION_H
#define INC_10_ATOMIC_OPERATION_REDUCTION_H

// @reduction_wrp_atmc_kernel.cu
// @reduction_blk_atmc_kernel.cu
// @reduction_kernel.cu
void atomic_reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#define min(a, b) (a) < (b) ? a : b

#endif //INC_10_ATOMIC_OPERATION_REDUCTION_H
