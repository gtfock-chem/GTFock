#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mkl.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#include "pdgemm.h"

static void copyMat(int m, int n, double *From, int ldfrom, double *To, int ldto)
{
    #pragma omp parallel for
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) To[r * ldto + c] = From[r * ldfrom + c];
}

static double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

static void ReduceTo2D(
    int myrow, int mycol, int mygrd,
    int nrows, int ncols, double *S, int ghost, MPI_Comm comm_3D
)
{
    if (mygrd != ghost)
    {
        if (mycol == mygrd)
        {
            int coords[3] = { myrow, mycol, 0 }, to;
            MPI_Cart_rank(comm_3D, coords, &to);
            MPI_Send(&S[0], nrows * ncols, MPI_DOUBLE, to, 0, comm_3D);
        }
    }
    else
    {
        if (mycol)
        {
            MPI_Status status;
            int coords[3] = { myrow, mycol, mycol }, from;
            MPI_Cart_rank(comm_3D, coords, &from);
            MPI_Recv(S, nrows * ncols, MPI_DOUBLE, from, 0, comm_3D, &status);
        }
    }
}

void ReduceToGrd0(
    int myrow, int mycol, int mygrd, 
    int S_len, double *S, MPI_Comm comm_3D
)
{
    // Process (myrow, myrow, mygrd) send to process (myrow, mygrd, 0)
    // Process (0, 0, 0) need not to send / recv
    if (myrow + mycol + mygrd >= 1)  
    {
        if (myrow == mycol) 
        {
            int coords[3] = {myrow, mygrd, 0}, dst;
            MPI_Cart_rank(comm_3D, coords, &dst);
            MPI_Send(S, S_len, MPI_DOUBLE, dst, 0, comm_3D);
        }
        
        if (mygrd == 0)
        {
            MPI_Status status;
            int coords[3] = {myrow, myrow, mycol}, src;
            MPI_Cart_rank(comm_3D, coords, &src);
            MPI_Recv(S, S_len, MPI_DOUBLE, src, 0, comm_3D, &status);
        }
    }
}

// Used by McWeeny purification only, input D, output D^2 and D^3
// All MPI_Comms used in this function are fixed, so we can duplicate 
// comm_row to further utilized the network bandwidth
int pdgemm3D(int myrow, int mycol, int mygrd,
             MPI_Comm comm_row, MPI_Comm comm_col,
             MPI_Comm comm_grd, MPI_Comm comm_3D,
             int *nr, int *nc,
             int nrows, int ncols,
             double *D_, double *D2_, double *D3_,
             tmpbuf_t *tmpbuf, double *dgemm_time)
{
    int ncols0 = nc[0], nrows0 = nr[0];
    if (dgemm_time != NULL) *dgemm_time = 0.0;
    assert(nrows  == nr[myrow] && ncols == nc[mycol]);
    assert(ncols0 >= ncols && nrows0 >= nrows);

    double *A   = tmpbuf->A;
    double *S   = tmpbuf->S;
    double *C   = tmpbuf->C;
    double *A_i = tmpbuf->A_i;
    double *S_i = tmpbuf->S_i;
    double *C_i = tmpbuf->C_i;
    double st, et;

    memset(A, 0, sizeof(double) * nrows0 * ncols0);
    copyMat(nrows, ncols, D_, ncols, A, ncols0);

    // 1.1. Replicate D matrix on all planes
    MPI_Bcast(A, nrows0 * ncols0, MPI_DOUBLE, 0, comm_grd);

    // 1.2. Broadcast A_i row
    if (myrow == mygrd) copyMat(nrows0, ncols0, A, ncols0, &A_i[0], ncols0);
    MPI_Bcast(&A_i[0], nrows0 * ncols0, MPI_DOUBLE, mygrd, comm_col);

    // 2.1. Do local dgemm
    st = get_wtime_sec();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols0, ncols0,
                ncols0, 1.0, A, ncols0, &A_i[0], ncols0, 0.0, &S_i[0], ncols0);
    et = get_wtime_sec();
    if (dgemm_time != NULL) *dgemm_time += et - st;
    
    // 2.2. Reduce S_i into a column i on row i
    MPI_Reduce(&S_i[0], S, nrows0 * ncols0, MPI_DOUBLE, MPI_SUM, myrow, comm_row);
    
    // 2.3. Copy S to S_i, ready to broadcast
    copyMat(nrows0, ncols0, &S[0], ncols0, &S_i[0], ncols0);
    
    // 3.1. Broadcast S_i
    MPI_Bcast(&S_i[0], nrows0 * ncols0, MPI_DOUBLE, mycol, comm_col);

    // 3.2. C_i=A*S_i
    st = get_wtime_sec();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ncols0, ncols0,
                ncols0, 1.0, A, ncols0, &S_i[0], ncols0, 0.0, &C_i[0], ncols0);
    et = get_wtime_sec();
    if (dgemm_time != NULL) *dgemm_time += et - st;
    
    // 3.3. Reduce C_i into a column on plane i
    MPI_Reduce(&C_i[0], C, nrows0 * ncols0, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);
    
    // 3.4. Reduce S to plane 0
    ReduceToGrd0(myrow, mycol, mygrd, nrows0 * ncols0, S, comm_3D);

    // 3.5. Reduce C to plane 0
    ReduceTo2D(myrow, mycol, mygrd, nrows0, ncols0, C, 0, comm_3D);

    // 4. Copy results to D2 and D3
    copyMat(nrows, ncols, S, ncols0, D2_, ncols);
    copyMat(nrows, ncols, C, ncols0, D3_, ncols);

    return 0;
}

// True parallel dgemm, input A, B return C := A * B
void pdgemm3D_2(int myrow, int mycol, int mygrd,
                MPI_Comm comm_row, MPI_Comm comm_col,
                MPI_Comm comm_grd, MPI_Comm comm_3D,
                int *nr, int *nc, int nrows, int ncols,
                double *A_block_, double *B_block_,
                double *C_block_, tmpbuf_t *tmpbuf, double *dgemm_time)
{
    int ncols0 = nc[0], nrows0 = nr[0];
    if (dgemm_time != NULL) *dgemm_time = 0.0;
    assert(nrows == nr[myrow] && ncols == nc[mycol]);
    assert(ncols0 >= ncols && nrows0 >= nrows);

    double *A_block = tmpbuf->A;
    double *B_block = tmpbuf->C;
    double *C_block = tmpbuf->S;
    double *B_block_copy = tmpbuf->A_i;
    double *C_i = tmpbuf->C_i;
    double st, et;
    
    memset(A_block, 0, sizeof (double) * nrows0 * ncols0);
    memset(B_block, 0, sizeof (double) * nrows0 * ncols0);
    copyMat(nrows, ncols, A_block_, ncols, A_block, ncols0);
    copyMat(nrows, ncols, B_block_, ncols, B_block, ncols0);

    // 1.1. Broadcast A blocks to each grid
    MPI_Bcast(A_block, nrows0 * ncols0, MPI_DOUBLE, 0, comm_grd);

    // 1.2. For matrix B at grid 0, send row i (except row 0) to grid i
    if (mygrd == 0 && myrow != 0) 
    {
        int coords[3] = { myrow, mycol, myrow }, to;
        MPI_Cart_rank(comm_3D, coords, &to);
        MPI_Send(&B_block[0], nrows0 * ncols0, MPI_DOUBLE, to, 0, comm_3D);
    }
    if (mygrd && myrow == mygrd) 
    {
        MPI_Status status;
        int coords[3] = { myrow, mycol, 0 }, from;
        MPI_Cart_rank(comm_3D, coords, &from);
        MPI_Recv(&B_block[0], nrows0 * ncols0, MPI_DOUBLE, from, 0, comm_3D, &status);
    }
    // 1.3. Spread / Bcast the row block of B_block on each grid 
    copyMat(nrows0, ncols0, &B_block[0], ncols0, &B_block_copy[0], ncols0);
    MPI_Bcast(&B_block_copy[0], nrows0 * ncols0, MPI_DOUBLE, mygrd, comm_col);

    // 2.1. Do local dgemm
    st = get_wtime_sec();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols0, ncols0,
                ncols0, 1.0, A_block, ncols0, &B_block_copy[0], ncols0, 0.0,
                &C_i[0], ncols0);
    et = get_wtime_sec();
    if (dgemm_time != NULL) *dgemm_time += et - st;

    // 2.2. Reduce C_i into a column i on plane i
    MPI_Reduce(&C_i[0], C_block, nrows0 * ncols0, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);

    // 2.3. Reduce C to plane 0
    ReduceTo2D(myrow, mycol, mygrd, nrows0, ncols0, C_block, 0, comm_3D);

    // 3. Copy result to C 
    copyMat(nrows, ncols, C_block, ncols0, C_block_, ncols);
}


void allocate_tmpbuf (int nrows, int ncols, int *nr, int *nc,
                      tmpbuf_t * tmpbuf)
{
    int ncols0 = nc[0], nrows0 = nr[0];
    assert (ncols0 >= ncols && nrows0 >= nrows);

    int block_size = nrows0 * ncols0;
    tmpbuf->A   = (double *) _mm_malloc(sizeof(double) * block_size * 6, 64);
    tmpbuf->S   = tmpbuf->A   + block_size;
    tmpbuf->C   = tmpbuf->S   + block_size;
    tmpbuf->A_i = tmpbuf->C   + block_size;
    tmpbuf->S_i = tmpbuf->A_i + block_size;
    tmpbuf->C_i = tmpbuf->S_i + block_size;
	

    #pragma omp parallel for schedule(static)
    #pragma simd
    for (int i = 0; i < nrows0 * ncols0; i++)
    {
        tmpbuf->A[i] = 0;
        tmpbuf->C[i] = 0;
        tmpbuf->S[i] = 0;
        tmpbuf->A_i[i] = 0;
        tmpbuf->C_i[i] = 0;
        tmpbuf->S_i[i] = 0;
    }
}


void dealloc_tmpbuf (tmpbuf_t * tmpbuf)
{
    _mm_free(tmpbuf->A);
}
