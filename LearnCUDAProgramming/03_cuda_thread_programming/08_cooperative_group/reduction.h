//
// Created by Zhou on 2023/3/26.
//

#ifndef INC_08_COOPERATIVE_GROUP_REDUCTION_H
#define INC_08_COOPERATIVE_GROUP_REDUCTION_H

// @reduction_kernel.cu
void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#define min(a, b) (a) < (b) ? (a) : (b)

#endif //INC_08_COOPERATIVE_GROUP_REDUCTION_H
