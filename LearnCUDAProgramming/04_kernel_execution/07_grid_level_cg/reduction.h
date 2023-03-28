//
// Created by Zhou on 2023/3/28.
//

#ifndef INC_07_GRID_LEVEL_CG_REDUCTION_H
#define INC_07_GRID_LEVEL_CG_REDUCTION_H

// @reduction_loop_kernel.cu
int reduction_grid_sync(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)

#endif //INC_07_GRID_LEVEL_CG_REDUCTION_H
