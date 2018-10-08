#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mkl.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <omp.h>

#include "pdgemm.h"

static void copyMat(int m, int n, double *From, int ldfrom, double *To, int ldto)
{
    #pragma omp parallel for
    for (int r = 0; r < m; r++)
        memcpy(To + r * ldto, From + r * ldfrom, sizeof(double) * n);
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

// src has nrows * ncols, dst has ncols * nrows
#define TRANS_BS 32
void myTranspose(double *src, double *dst, int nrows, int ncols)
{
    int nrb = (nrows + TRANS_BS - 1) / TRANS_BS;
    int ncb = (ncols + TRANS_BS - 1) / TRANS_BS;
    
    #pragma omp parallel for
    for (int ib = 0; ib < nrb * ncb; ib++)
    {
        int irow0 = (ib / ncb) * TRANS_BS;
        int icol0 = (ib % ncb) * TRANS_BS;
        int irow1 = irow0 + TRANS_BS;
        int icol1 = icol0 + TRANS_BS;
        if (irow1 > nrows) irow1 = nrows;
        if (icol1 > ncols) icol1 = ncols;
        
        for (int icol = icol0; icol < icol1; icol++)
        {
            #pragma simd
            for (int irow = irow0; irow < irow1; irow++)
            {
                int src_idx  = irow * ncols + icol;
                int dst_idx  = icol * nrows + irow;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

#define N_DUP    4
#define N_DUP_05 2
MPI_Comm    comm_rows[N_DUP], comm_cols[N_DUP], comm_grds[N_DUP], comm_3Ds[N_DUP];
MPI_Status  status[N_DUP];
MPI_Request reqs[N_DUP], reqs0[N_DUP];
int spos[N_DUP + 1], blklen[N_DUP];
int row_spos[N_DUP + 1], row_blklen[N_DUP];
int comm_dupped = 0;

static void dup_comms(MPI_Comm comm_row, MPI_Comm comm_col, MPI_Comm comm_grd, MPI_Comm comm_3D, int nrow, int ncol)
{
    if (comm_dupped == 1) return;
    comm_dupped = 1;
    
    int remainder  = nrow % N_DUP;
    int block_size = nrow / N_DUP;
    for (int i = 0; i < remainder; i++)
    {
        row_blklen[i] = block_size + 1;
        blklen[i] = row_blklen[i] * ncol;
    }
    for (int i = remainder; i < N_DUP; i++)
    {
        row_blklen[i] = block_size;
        blklen[i] = row_blklen[i] * ncol;
    }
    
    
    row_spos[0] = spos[0] = 0;
    for (int i = 0; i < N_DUP; i++)
    {
        MPI_Comm_dup(comm_row, &comm_rows[i]);
        MPI_Comm_dup(comm_col, &comm_cols[i]);
        MPI_Comm_dup(comm_grd, &comm_grds[i]);
        MPI_Comm_dup(comm_3D , &comm_3Ds [i]);
        spos[i + 1] = spos[i] + blklen[i];
        row_spos[i + 1] = row_spos[i] + row_blklen[i];
    }
}

static void Symmtrize_D2_Bcast(int myrow, int mycol, int mygrd, double *S, MPI_Comm comm_3D)
{
    if (myrow >= mygrd)
    {
        MPI_Request req;
        int coords[3] = {mygrd, mygrd, myrow}, dst;
        MPI_Cart_rank(comm_3D, coords, &dst);
        for (int i = 0; i < N_DUP; i++)
        {
            // Wait the i-th block reduction to be finished
            MPI_Wait(&reqs[i], &status[i]);
            
            // Strict lower triangle processes send the reduced i-th block 
            // to its symmetric position
            if (myrow > mygrd) MPI_Isend(S + spos[i], blklen[i], MPI_DOUBLE, dst, 0, comm_3Ds[i], &req);
            
            // Broadcast the reduced block
            MPI_Ibcast(S + spos[i], blklen[i], MPI_DOUBLE, mycol, comm_cols[i], &reqs[i]);
        }
    }
    
    if (myrow < mygrd)
    {
        int coords[3] = {mygrd, mygrd, myrow}, src;
        MPI_Cart_rank(comm_3D, coords, &src);
        
        // Strict upper triangle processes receive i-th reduced block
        // from its symmetric position
        for (int i = 0; i < N_DUP; i++)
            MPI_Irecv(S + spos[i], blklen[i], MPI_DOUBLE, src, 0, comm_3Ds[i], &reqs[i]);
        
        // Broadcast the reduced block
        for (int i = 0; i < N_DUP; i++)
        {
            // Wait the i-th block to be received
            MPI_Wait(&reqs[i], &status[i]);
            
            // Broadcast the reduced block
            MPI_Ibcast(S + spos[i], blklen[i], MPI_DOUBLE, mycol, comm_cols[i], &reqs[i]);
        }
    }
}

static void ReduceToGrd0(
    int myrow, int mycol, int mygrd, int nrows, int ncols, 
    double *S, double *S_buf, MPI_Comm comm_3D
)
{
    // Process (myrow, myrow, mygrd) send to process (myrow, mygrd, 0)
    // Process (0, 0, 0) need not to send / recv
    if (myrow + mycol + mygrd >= 1)  
    {
        if (myrow == mycol) 
        {
            MPI_Request req;
            int coords[3] = {myrow, mygrd, 0}, dst;
            MPI_Cart_rank(comm_3D, coords, &dst);
            if (myrow < mygrd)
            {
                // Upper triangle blocks are received from lower triangle, need to be transposed
                myTranspose(S, S_buf, ncols, nrows);
                for (int i = 0; i < N_DUP; i++)
                    MPI_Isend(S_buf + spos[i], blklen[i], MPI_DOUBLE, dst, 0, comm_3Ds[i], &req);
            } else {
                for (int i = 0; i < N_DUP; i++)
                    MPI_Isend(S     + spos[i], blklen[i], MPI_DOUBLE, dst, 0, comm_3Ds[i], &req);
            }
        }
        
        if (mygrd == 0)
        {
            int coords[3] = {myrow, myrow, mycol}, src;
            MPI_Cart_rank(comm_3D, coords, &src);
            if (myrow == mycol) 
            {
                for (int i = 0; i < N_DUP; i++)
                    MPI_Irecv(S_buf + spos[i], blklen[i], MPI_DOUBLE, src, 0, comm_3Ds[i], &reqs0[i]);
            } else {
                for (int i = 0; i < N_DUP; i++)
                    MPI_Irecv(S     + spos[i], blklen[i], MPI_DOUBLE, src, 0, comm_3Ds[i], &reqs0[i]);
            }
            MPI_Waitall(N_DUP, &reqs0[0], &status[0]);
        }
    }
}

static void ReduceTo2D_symm(
    int myrow, int mycol, int mygrd, int nrows, int ncols, 
    double *S, double *S_buf, MPI_Comm comm_3D, MPI_Comm comm_grd
)
{
    // Source processes, to send
    if (mycol == mygrd && myrow >= mygrd)
    {
        if (mycol == 0 && myrow == 0)
            MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        
        // (i, 0, 0) --> (0, i, 0) where i > 0
        if (mycol == 0 && myrow > 0)
        {
            int coords[3] = {0, myrow, 0}, dst;
            MPI_Cart_rank(comm_3D, coords, &dst);
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Isend(S + spos[i], blklen[i], MPI_DOUBLE, dst, 0, comm_3Ds[i], &reqs[i]);
            }
        }
        
        // (i, i, i) --> (i, i, 0) where i > 0
        if (myrow == mycol && myrow > 0)
        {
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Isend(S + spos[i], blklen[i], MPI_DOUBLE, 0, 0, comm_grds[i], &reqs[i]);
            }
        }
        
        // (i, j, j) --> (i, j, 0) && (j, i, 0) where i > j & j > 0
        if (myrow > mycol && mycol > 0)
        {
            int coords[3] = {mycol, myrow, 0}, dst;
            MPI_Cart_rank(comm_3D, coords, &dst);
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Isend(S + spos[i], blklen[i], MPI_DOUBLE, 0,   0, comm_grds[i], &reqs[i]);
                MPI_Isend(S + spos[i], blklen[i], MPI_DOUBLE, dst, 0, comm_3Ds[i],  &reqs[i]);
            }
        }
    }
    
    // Destination processes, to receive
    if (mygrd == 0 && mycol > 0)
    {
        // (0, j, 0) <-- (j, 0, 0) where j > 0
        if (myrow == 0)
        {
            int coords[3] = {mycol, 0, 0}, src;
            MPI_Cart_rank(comm_3D, coords, &src);
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Irecv(S_buf + spos[i], blklen[i], MPI_DOUBLE, src, 0, comm_3Ds[i], &reqs[i]);
            }
            MPI_Waitall(N_DUP, &reqs[0], &status[0]);
            myTranspose(S_buf, S, ncols, nrows);
        }
        
        // (i, j, 0) <-- (i, j, j) where i >= j && j > 0
        if (myrow >= mycol)
        {
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Irecv(S + spos[i], blklen[i], MPI_DOUBLE, mycol, 0, comm_grds[i], &reqs[i]);
            }
            MPI_Waitall(N_DUP, &reqs[0], &status[0]);
        }
        
        // (i, j, 0) <-- (j, i, i) where j > i, i > 0
        if (mycol > myrow && myrow > 0)
        {
            int coords[3] = {mycol, myrow, myrow}, src;
            MPI_Cart_rank(comm_3D, coords, &src);
            for (int i = 0; i < N_DUP; i++)
            {
                MPI_Wait(&reqs[i], &status[i]);
                MPI_Irecv(S_buf + spos[i], blklen[i], MPI_DOUBLE, src, 0, comm_3Ds[i], &reqs[i]);
            }
            MPI_Waitall(N_DUP, &reqs[0], &status[0]);
            myTranspose(S_buf, S, ncols, nrows);
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
    assert(nrows  == nr[myrow] && ncols == nc[mycol]);
    assert(ncols0 >= ncols && nrows0 >= nrows);
    
    dup_comms(comm_row, comm_col, comm_grd, comm_3D, nrows0, ncols0);

    double *A   = tmpbuf->A;
    double *S   = tmpbuf->S;
    double *C   = tmpbuf->C;
    double *A_i = tmpbuf->A_i;
    double *S_i = tmpbuf->S_i;
    double *C_i = tmpbuf->C_i;
    double st, et, _dgemm_time = 0.0;

    #pragma omp parallel for 
    for (int r = 0; r < nrows0; r++)
    {
        if (r < nrows)
        {
            memcpy(A + r * ncols0, D_ + r * ncols, sizeof(double) * ncols);
            memset(A + r * ncols0 + ncols, 0, sizeof(double) * (ncols0 - ncols));
        } else {
            memset(A + r * ncols0, 0, sizeof(double) * ncols0);
        }
    }

    // 1.1. Replicate D matrix on all planes
    //MPI_Bcast(A, nrows0 * ncols0, MPI_DOUBLE, 0, comm_grd);
    for (int i = 0; i < N_DUP; i++)
        MPI_Ibcast(A + spos[i], blklen[i], MPI_DOUBLE, 0, comm_grds[i], &reqs[i]);
    //MPI_Waitall(N_DUP, &reqs[0], &status[0]);

    // 1.2. Broadcast A_i row
    if (myrow == mygrd) A_i = A;  // Need not to copy, we won't modify the data later
    if (myrow == mygrd && myrow > 0)
    {
        for (int i = 0; i < N_DUP; i++)
        {
            // When the i-th block of A is broadcast, Ibcast it as A_i immediately
            MPI_Wait(&reqs[i], &status[i]);  
            MPI_Ibcast(A_i + spos[i], blklen[i], MPI_DOUBLE, mygrd, comm_cols[i], &reqs0[i]);
        }
    } else {
        // Start the Ibcast of B without waiting
        for (int i = 0; i < N_DUP; i++)
            MPI_Ibcast(A_i + spos[i], blklen[i], MPI_DOUBLE, mygrd, comm_cols[i], &reqs0[i]);
        // Wait the broadcast of A to complete
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);
    }
    MPI_Waitall(N_DUP, &reqs0[0], &status[0]);
    
    // 2.1 Do local dgemm
    st = get_wtime_sec();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols0, ncols0,
                ncols0, 1.0, A, ncols0, A_i, ncols0, 0.0, S_i, ncols0);
    et = get_wtime_sec();
    _dgemm_time += et - st;

    // 2.2. Reduce S_i into a column i on row i
    //MPI_Reduce(&S_i[0], S, nrows0 * ncols0, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);
    if (myrow >= mygrd)
    {
        for (int i = 0; i < N_DUP; i++)
            MPI_Ireduce(&S_i[spos[i]], S + spos[i], blklen[i], MPI_DOUBLE, MPI_SUM, myrow, comm_rows[i], &reqs[i]);
    }
    //MPI_Waitall(N_DUP, &reqs[0], &status[0]);
    
    // 3.1. Copy S to S_i, ready to broadcast
    //S_i = S;  // Need not to copy, won't affect ReduceToGrd0
    
    // 3.2. Broadcast S_i
    if (myrow == mycol)
    {
        // Strict lower triangle part send reduced S to strict upper triangle part
        // myrow==mycol plane broadcast the S as root
        Symmtrize_D2_Bcast(myrow, mycol, mygrd, S, comm_3D);
    } else {
        // Receive the broadcast S
        for (int i = 0; i < N_DUP; i++)
        {
            MPI_Wait(&reqs[i], &status[i]);
            MPI_Ibcast(S + spos[i], blklen[i], MPI_DOUBLE, mycol, comm_cols[i], &reqs[i]);
        }
    }
    MPI_Waitall(N_DUP, &reqs[0], &status[0]);
    
    // 3.3 C_i=A*S_i
    st = get_wtime_sec();
    if (mycol >= mygrd)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ncols0, ncols0,
                    ncols0, 1.0, A, ncols0, S, ncols0, 0.0, C_i, ncols0);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols0, ncols0,
                    ncols0, 1.0, A, ncols0, S, nrows0, 0.0, C_i, ncols0);
    }
    et = get_wtime_sec();
    _dgemm_time += et - st;

    // 3.4. Reduce C_i into a column on plane i
    //MPI_Reduce(&C_i[0], C, nrows0 * ncols0, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);
    if (myrow >= mygrd)
    {
        for (int i = 0; i < N_DUP; i++)
            MPI_Ireduce(&C_i[spos[i]], C + spos[i], blklen[i], MPI_DOUBLE, MPI_SUM, mygrd, comm_rows[i], &reqs[i]);
    }
    
    // 4.1. Reduce S to plane 0
    ReduceToGrd0(myrow, mycol, mygrd, nrows0, ncols0, S, S_i, comm_3D);
    if (mygrd == 0 && myrow == mycol && myrow > 0) S = S_i;

    // 4.2. Reduce C to plane 0
    ReduceTo2D_symm(myrow, mycol, mygrd, nrows0, ncols0, C, C_i, comm_3D, comm_grd);

    // 4.3. Copy results to D2 and D3
    if (mygrd == 0)
    {
        size_t row_size = ncols * sizeof(double);
        #pragma omp parallel for
        for (int irow = 0; irow < nrows; irow++)
        {
            memcpy(D2_ + ncols * irow, S + ncols0 * irow, row_size);
            memcpy(D3_ + ncols * irow, C + ncols0 * irow, row_size);
        }
    }
    
    if (dgemm_time != NULL) *dgemm_time = _dgemm_time;

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


void allocate_tmpbuf(int nrows, int ncols, int *nr, int *nc, tmpbuf_t *tmpbuf)
{
    int ncols0 = nc[0], nrows0 = nr[0];
    assert (ncols0 >= ncols && nrows0 >= nrows);

    int block_size_align64b = (nrows0 * ncols0 + 7) / 8 * 8;
    tmpbuf->A   = (double *) _mm_malloc(sizeof(double) * block_size_align64b * 6, 64);
    tmpbuf->S   = tmpbuf->A   + block_size_align64b;
    tmpbuf->C   = tmpbuf->S   + block_size_align64b;
    tmpbuf->A_i = tmpbuf->C   + block_size_align64b;
    tmpbuf->S_i = tmpbuf->A_i + block_size_align64b;
    tmpbuf->C_i = tmpbuf->S_i + block_size_align64b;

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


void dealloc_tmpbuf(tmpbuf_t *tmpbuf)
{
    _mm_free(tmpbuf->A);
}