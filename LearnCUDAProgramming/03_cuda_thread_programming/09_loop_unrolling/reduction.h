//
// Created by Zhou on 2023/3/26.
//

#ifndef INC_09_LOOP_UNROLLING_REDUCTION_H
#define INC_09_LOOP_UNROLLING_REDUCTION_H

// @reduction_kernel.cu
void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#endif //INC_09_LOOP_UNROLLING_REDUCTION_H
