#ifndef __FOCK_TASK_H__
#define __FOCK_TASK_H__

#include "pfock.h"
#include "CInt.h"

void init_block_buf(BasisSet_t _basis, PFock_t pfock);

void fock_task(
    int nblks_col, int sblk_row, int sblk_col, 
    int task, int startrow, int startcol, int repack_D
);

void reset_F(int numF, int num_dmat, double *F1, double *F2, double *F3, int sizeX1, int sizeX2, int sizeX3);

void reduce_F(double *F1, double *F2, double *F3, int maxrowsize, int maxcolsize, int ldX3, int ldX4, int ldX5, int ldX6);


#endif /* #define __FOCK_TASK_H__ */
