#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//#include <ga.h>
//#include <macdecls.h>
#include <malloc.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <assert.h>
#include <math.h>

#include "pfock.h"
#include "config.h"
#include "fock_task.h"
#include "fock_buf.h"
#include "taskq.h"
#include "screening.h"
#include "one_electron.h"

#include "GTMatrix.h"
#include "utils.h"

static PFockStatus_t init_fock(PFock_t pfock)
{
    int nshells;
    int nprow;
    int npcol;
    int i;
    int j;
    int n0;
    int n1;
    int t;
    int n2;
    int myrank;
    int nbp_row;
    int nbp_col;
    int nbp_p;
    int nshells_p;
        
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    nbp_p = pfock->nbp_p;
    nbp_row = pfock->nprow * nbp_p;
    nbp_col = pfock->npcol *nbp_p;
    nshells = pfock->nshells;
    // partition task blocks
    nprow = pfock->nprow;
    npcol = pfock->npcol;
    pfock->rowptr_f = (int *)PFOCK_MALLOC(sizeof(int) * (nprow + 1));
    pfock->colptr_f = (int *)PFOCK_MALLOC(sizeof(int) * (npcol + 1));
    pfock->rowptr_sh = (int *)PFOCK_MALLOC(sizeof(int) * (nprow + 1));
    pfock->colptr_sh = (int *)PFOCK_MALLOC(sizeof(int) * (npcol + 1));
    pfock->rowptr_blk = (int *)PFOCK_MALLOC(sizeof(int) * (nprow + 1));
    pfock->colptr_blk = (int *)PFOCK_MALLOC(sizeof(int) * (npcol + 1));
    if (NULL == pfock->rowptr_f || NULL == pfock->colptr_f ||
        NULL == pfock->rowptr_sh || NULL == pfock->colptr_sh ||
        NULL == pfock->rowptr_blk || NULL == pfock->colptr_blk)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    pfock->mem_cpu += 3.0 * sizeof(int) * ((nprow + 1) + (npcol + 1));
    // for row partition
    n0 = nshells/nprow;
    t = nshells%nprow;
    n1 = (nshells + nprow - 1)/nprow;    
    n2 = n1 * t;
    for (i = 0; i < nprow; i++)
    {
        pfock->rowptr_blk[i] = nbp_p * i;
        pfock->rowptr_sh[i] = i < t ? n1 * i : n2 + (i - t) * n0;
        pfock->rowptr_f[i] = pfock->f_startind[pfock->rowptr_sh[i]];
    }
    pfock->rowptr_blk[i] = nbp_row;
    pfock->rowptr_sh[i] = nshells;
    pfock->rowptr_f[i] = pfock->nbf;
    // set own
    pfock->sblk_row = pfock->rowptr_blk[myrank/npcol];
    pfock->eblk_row = pfock->rowptr_blk[myrank/npcol + 1] - 1;
    pfock->nblks_row = pfock->eblk_row - pfock->sblk_row + 1;    
    pfock->sshell_row = pfock->rowptr_sh[myrank/npcol];
    pfock->eshell_row = pfock->rowptr_sh[myrank/npcol + 1] - 1;
    pfock->nshells_row = pfock->eshell_row - pfock->sshell_row + 1;    
    pfock->sfunc_row = pfock->rowptr_f[myrank/npcol];
    pfock->efunc_row = pfock->rowptr_f[myrank/npcol + 1] - 1;
    pfock->nfuncs_row = pfock->efunc_row - pfock->sfunc_row + 1;   
    // for col partition
    n0 = nshells/npcol;
    t = nshells%npcol;
    n1 = (nshells + npcol - 1)/npcol;    
    n2 = n1 * t;
    for (i = 0; i < npcol; i++)
    {
        pfock->colptr_blk[i] = nbp_p * i;
        pfock->colptr_sh[i] = i < t ? n1 * i : n2 + (i - t) * n0;
        pfock->colptr_f[i] = pfock->f_startind[pfock->colptr_sh[i]];
    }
    pfock->colptr_blk[i] = nbp_col;
    pfock->colptr_sh[i] = nshells;
    pfock->colptr_f[i] = pfock->nbf;    
    // set own
    pfock->sblk_col = pfock->colptr_blk[myrank%npcol];
    pfock->eblk_col = pfock->colptr_blk[myrank%npcol + 1] - 1;
    pfock->nblks_col = pfock->eblk_col - pfock->sblk_col + 1;    
    pfock->sshell_col = pfock->colptr_sh[myrank%npcol];
    pfock->eshell_col = pfock->colptr_sh[myrank%npcol + 1] - 1;
    pfock->nshells_col = pfock->eshell_col - pfock->sshell_col + 1;    
    pfock->sfunc_col = pfock->colptr_f[myrank%npcol];
    pfock->efunc_col = pfock->colptr_f[myrank%npcol + 1] - 1;
    pfock->nfuncs_col = pfock->efunc_col - pfock->sfunc_col + 1;
     
    pfock->ntasks = nbp_p * nbp_p;
    pfock->blkrowptr_sh = (int *)PFOCK_MALLOC(sizeof(int) * (nbp_row + 1));
    pfock->blkcolptr_sh = (int *)PFOCK_MALLOC(sizeof(int) * (nbp_col + 1));
    pfock->mem_cpu += sizeof(int) * ((nbp_row + 1) + (nbp_col + 1));
    if (NULL == pfock->blkrowptr_sh || NULL == pfock->blkcolptr_sh)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }

    // tasks 2D partitioning
    // row
    for (i = 0; i < nprow; i++)
    {
        nshells_p = pfock->rowptr_sh[i + 1] - pfock->rowptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            pfock->blkrowptr_sh[i *nbp_p + j] = pfock->rowptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkrowptr_sh[i * nbp_p] = nshells;
    // col
    for (i = 0; i < npcol; i++)
    {
        nshells_p = pfock->colptr_sh[i + 1] - pfock->colptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            pfock->blkcolptr_sh[i *nbp_p + j] = pfock->colptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkcolptr_sh[i * nbp_p] = nshells;
 
    return PFOCK_STATUS_SUCCESS;
}


static void recursive_bisection (int *rowptr, int first, int last,
                                 int npartitions, int *partition_ptr)
{
    int offset = rowptr[first];
    int nnz = rowptr[last] - rowptr[first];

    if(npartitions == 1)
    {
        partition_ptr[0] = first;
        return;
    }

    int left = npartitions/2;
    double ideal = ((double)nnz * (double)left)/npartitions;
    int i;
    for(i = first; i < last; i++)
    {
        double count = rowptr[i] - offset;
        double next_count = rowptr[i + 1] - offset;
        if(next_count > ideal)
        {
            if(next_count - ideal > ideal - count)
            {
                recursive_bisection(rowptr, first, i, left, partition_ptr);
                recursive_bisection(rowptr, i, last,
                                    npartitions - left, partition_ptr + left);
                return;
            }
            else
            {
                recursive_bisection(rowptr, first, i + 1, left, partition_ptr);
                recursive_bisection(rowptr, i + 1, last,
                                    npartitions - left, partition_ptr + left);
                return;
            }
        }
    }
}


static int nnz_partition (int m, int nnz, int min_nrows,
                          int *rowptr, int npartitions, int *partition_ptr)
{
    recursive_bisection(rowptr, 0, m, npartitions, partition_ptr);
    partition_ptr[npartitions] = m;

    for (int i = 0; i < npartitions; i++)
    {
        int nrows = partition_ptr[i + 1] - partition_ptr[i];
        if (nrows < min_nrows)
        {
            return -1;
        }
    }
    
    return 0;
}


static PFockStatus_t repartition_fock (PFock_t pfock)
{
    int nshells = pfock->nshells;
    int nnz = pfock->nnz;
    int nbp_p = pfock->nbp_p;
    int nprow = pfock->nprow;
    int npcol = pfock->npcol;
    int *shellptr = pfock->shellptr;
    int myrank;
    int ret;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

    // for row partition
    int *newrowptr = (int *)malloc (sizeof(int) * (nprow + 1));
    int *newcolptr = (int *)malloc (sizeof(int) * (npcol + 1));
    ret = nnz_partition (nshells, nnz, nbp_p, shellptr, nprow, newrowptr);    
    if (ret != 0)
    {
        PFOCK_PRINTF (1, "nbp_p is too large\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    ret = nnz_partition (nshells, nnz, nbp_p, shellptr, npcol, newcolptr);
    if (ret != 0)
    {
        PFOCK_PRINTF (1, "nbp_p is too large\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    memcpy (pfock->rowptr_sh, newrowptr, sizeof(int) * (nprow + 1));    
    memcpy (pfock->colptr_sh, newcolptr, sizeof(int) * (npcol + 1));
    free (newrowptr);
    free (newcolptr);
    
    for (int i = 0; i < nprow; i++)
    {
        pfock->rowptr_f[i] = pfock->f_startind[pfock->rowptr_sh[i]];
    }
    pfock->rowptr_f[nprow] = pfock->nbf;
    // set own  
    pfock->sshell_row = pfock->rowptr_sh[myrank/npcol];
    pfock->eshell_row = pfock->rowptr_sh[myrank/npcol + 1] - 1;
    pfock->nshells_row = pfock->eshell_row - pfock->sshell_row + 1;    
    pfock->sfunc_row = pfock->rowptr_f[myrank/npcol];
    pfock->efunc_row = pfock->rowptr_f[myrank/npcol + 1] - 1;
    pfock->nfuncs_row = pfock->efunc_row - pfock->sfunc_row + 1;  
    
    // for col partition
    for (int i = 0; i < npcol; i++)
    {
        pfock->colptr_f[i] = pfock->f_startind[pfock->colptr_sh[i]];
    }
    pfock->colptr_f[npcol] = pfock->nbf;    
    // set own   
    pfock->sshell_col = pfock->colptr_sh[myrank%npcol];
    pfock->eshell_col = pfock->colptr_sh[myrank%npcol + 1] - 1;
    pfock->nshells_col = pfock->eshell_col - pfock->sshell_col + 1;    
    pfock->sfunc_col = pfock->colptr_f[myrank%npcol];
    pfock->efunc_col = pfock->colptr_f[myrank%npcol + 1] - 1;
    pfock->nfuncs_col = pfock->efunc_col - pfock->sfunc_col + 1;
     
    // tasks 2D partitioning
    // row
    int nshells_p;
    int n0;
    int t;
    int n1;
    int n2;
    for (int i = 0; i < nprow; i++)
    {
        nshells_p = pfock->rowptr_sh[i + 1] - pfock->rowptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (int j = 0; j < nbp_p; j++)
        {
            pfock->blkrowptr_sh[i *nbp_p + j] = pfock->rowptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkrowptr_sh[nprow * nbp_p] = nshells;
    // col
    for (int i = 0; i < npcol; i++)
    {
        nshells_p = pfock->colptr_sh[i + 1] - pfock->colptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (int j = 0; j < nbp_p; j++)
        {
            pfock->blkcolptr_sh[i *nbp_p + j] = pfock->colptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkcolptr_sh[npcol * nbp_p] = nshells;

    // for correct_F
    pfock->FT_block = (double *)PFOCK_MALLOC(sizeof(double) *
        pfock->nfuncs_row * pfock->nfuncs_col);
    pfock->mem_cpu += 1.0 * pfock->nfuncs_row * pfock->nfuncs_col * sizeof(double);
    if (NULL == pfock->FT_block)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    return PFOCK_STATUS_SUCCESS;
}


static PFockStatus_t create_GA (PFock_t pfock)
{
    // Create global arrays
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    GTMatrix_t *gtm_ptrs[8];
    gtm_ptrs[0] = &pfock->gtm_Dmat;
    gtm_ptrs[1] = &pfock->gtm_Fmat;
    gtm_ptrs[2] = &pfock->gtm_Kmat;
    gtm_ptrs[3] = &pfock->gtm_Hmat;
    gtm_ptrs[4] = &pfock->gtm_Smat;
    gtm_ptrs[5] = &pfock->gtm_Xmat;
    gtm_ptrs[6] = &pfock->gtm_tmp1;
    gtm_ptrs[7] = &pfock->gtm_tmp2;
    for (int i = 0; i < 8; i++)
    {
        GTM_create(
            gtm_ptrs[i], MPI_COMM_WORLD, MPI_DOUBLE, 8,
            my_rank, pfock->nbf, pfock->nbf, 
            pfock->nprow, pfock->npcol,
            pfock->rowptr_f, pfock->colptr_f
        );
    }

    return PFOCK_STATUS_SUCCESS;
}


static void destroy_GA(PFock_t pfock)
{ 
    GTM_destroy(pfock->gtm_Dmat);
    GTM_destroy(pfock->gtm_Fmat);
    GTM_destroy(pfock->gtm_Kmat);
}


static PFockStatus_t create_FD_GArrays (PFock_t pfock)
{
    int sizeD1 = pfock->sizeX1;
    int sizeD2 = pfock->sizeX2;
    int sizeD3 = pfock->sizeX3;  
    
    // Create each process's F1, F2, F3 buffer matrix
    int *map = (int*) malloc(sizeof(int) * (3 + pfock->nprocs));
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for (int i = 0; i <= pfock->nprocs; i++) map[i] = i;
    map[pfock->nprocs + 1] = 0;
    map[pfock->nprocs + 2] = sizeD1;
    GTM_create(
        &pfock->gtm_F1, MPI_COMM_WORLD, MPI_DOUBLE, 8,
        my_rank, pfock->nprocs, sizeD1, 
        pfock->nprocs, 1,
        &map[0], &map[pfock->nprocs + 1]
    );
    map[pfock->nprocs + 2] = sizeD2;
    GTM_create(
        &pfock->gtm_F2, MPI_COMM_WORLD, MPI_DOUBLE, 8,
        my_rank, pfock->nprocs, sizeD2, 
        pfock->nprocs, 1,
        &map[0], &map[pfock->nprocs + 1]
    );
    map[pfock->nprocs + 2] = sizeD3;
    GTM_create(
        &pfock->gtm_F3, MPI_COMM_WORLD, MPI_DOUBLE, 8,
        my_rank, pfock->nprocs, sizeD3, 
        pfock->nprocs, 1,
        &map[0], &map[pfock->nprocs + 1]
    );
    free(map);
    
    pfock->getFockMatBufSize = 0;
    pfock->getFockMatBuf = NULL;

    return PFOCK_STATUS_SUCCESS; 
}


static PFockStatus_t create_buffers (PFock_t pfock)
{
    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    int myrow = myrank/pfock->npcol;
    int mycol = myrank%pfock->npcol;   
    int *ptrrow = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nshells);
    int *ptrcol = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nshells);
    if (NULL == ptrrow ||
        NULL == ptrcol) {
        PFOCK_PRINTF(1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;    
    }    

    // compute rowptr/pos and colptr/pos
    pfock->rowpos = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nshells);
    pfock->colpos = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nshells);
    pfock->rowptr = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nnz);
    pfock->colptr = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nnz);
    pfock->rowsize = (int *)PFOCK_MALLOC(sizeof(int) * pfock->nprow);
    pfock->colsize = (int *)PFOCK_MALLOC(sizeof(int) * pfock->npcol);
    pfock->mem_cpu += 1.0 * sizeof(int) *
        (2.0 * pfock->nshells + 2.0 * pfock->nnz +
         pfock->nprow + pfock->npcol);
    if (NULL == pfock->rowpos  ||
        NULL == pfock->colpos  ||
        NULL == pfock->rowptr  || 
        NULL == pfock->colptr  ||
        NULL == pfock->rowsize ||
        NULL == pfock->colsize) {
        PFOCK_PRINTF(1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;    
    }   
    int count = 0;
    int maxrowsize = 0;
    int maxrowfuncs = 0;
    for (int i = 0; i < pfock->nprow; i++) {
        compute_FD_ptr (pfock,
                        pfock->rowptr_sh[i], pfock->rowptr_sh[i+1] - 1,
                        ptrrow, &(pfock->rowsize[i]));
        maxrowsize =
            pfock->rowsize[i] > maxrowsize ? pfock->rowsize[i] : maxrowsize;
        int nfuncs = pfock->rowptr_f[i + 1] - pfock->rowptr_f[i];
        maxrowfuncs = nfuncs > maxrowfuncs ? nfuncs : maxrowfuncs;
        if (i == myrow) {
            pfock->sizemyrow = pfock->rowsize[i];
            init_FD_load(pfock, ptrrow,
                         &(pfock->loadrow), &(pfock->sizeloadrow));  
        }
        for (int j = pfock->rowptr_sh[i]; j < pfock->rowptr_sh[i+1]; j++) {
            pfock->rowpos[j] = ptrrow[j];
            for (int k = pfock->shellptr[j]; k < pfock->shellptr[j+1]; k++)
            {
                int sh = pfock->shellid[k];
                pfock->rowptr[count] = ptrrow[sh];
                count++;
            }
        }
    }
    count = 0;
    int maxcolsize = 0;
    int maxcolfuncs = 0;
    for (int i = 0; i < pfock->npcol; i++) {
        compute_FD_ptr (pfock,
                        pfock->colptr_sh[i], pfock->colptr_sh[i+1] - 1,
                        ptrcol, &(pfock->colsize[i]));
        maxcolsize =
            pfock->colsize[i] > maxcolsize ? pfock->colsize[i] : maxcolsize;
        int nfuncs = pfock->colptr_f[i + 1] - pfock->colptr_f[i];
        maxcolfuncs = nfuncs > maxcolfuncs ? nfuncs : maxcolfuncs;
        if (i == mycol) {
            pfock->sizemycol = pfock->colsize[i];
            init_FD_load (pfock, ptrcol,
                          &(pfock->loadcol), &(pfock->sizeloadcol));  
        }
        for (int j = pfock->colptr_sh[i]; j < pfock->colptr_sh[i+1]; j++) {
            pfock->colpos[j] = ptrcol[j];
            for (int k = pfock->shellptr[j]; k < pfock->shellptr[j+1]; k++) {
                int sh = pfock->shellid[k];
                pfock->colptr[count] = ptrcol[sh];
                count++;
            }
        }
    }
    PFOCK_FREE(ptrrow);
    PFOCK_FREE(ptrcol);
    pfock->maxrowsize = maxrowsize;
    pfock->maxcolsize = maxcolsize;
    pfock->maxrowfuncs = maxrowfuncs;
    pfock->maxcolfuncs = maxcolfuncs;
    int sizeX1 = maxrowfuncs * maxrowsize;
    int sizeX2 = maxcolfuncs * maxcolsize;
    int sizeX3 = maxrowsize * maxcolsize;
    pfock->sizeX1 = sizeX1;
    pfock->sizeX2 = sizeX2;
    pfock->sizeX3 = sizeX3;
    if (myrank == 0) {
        printf("  FD size (%d %d %d %d)\n",
            maxrowfuncs, maxcolfuncs, maxrowsize, maxcolsize);
    }
    
    // D buf
    size_t nbf2 = pfock->nbf * pfock->nbf;
    pfock->D_mat = (double*) PFOCK_MALLOC(sizeof(double) * nbf2);
    pfock->mem_cpu += 1.0 * sizeof(double) * nbf2;
    if (pfock->D_mat == NULL) 
    {
        PFOCK_PRINTF(1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    if (myrank == 0) printf("D1, D2, D3 size = %d, Dmat size = %lu\n", sizeX1 + sizeX2 + sizeX3, nbf2);

    
    // F buf
    int nthreads = pfock->nthreads;
    char *ncpu_str = getenv("nCPU_F");
    int ncpu_f;
    if (ncpu_str == NULL) {
        ncpu_f = 1;
    } else {
        ncpu_f = atoi(ncpu_str);
        if (ncpu_f <= 0 || ncpu_f > nthreads) {
            ncpu_f = 1;
        }
    }
    
    // We don't need multiple copies of F1, F2, F4, F5, F6 now, so just let 
    // numF = 1 here. If we set ncpu_f and numF according to the environment 
    // variable but only allocate using numF = 1, the program will crash.
    // We will get the environment variable later again.
    ncpu_f = nthreads; 
    
    int sizeX4 = maxrowfuncs * maxcolfuncs;
    int sizeX6 = maxrowsize  * maxcolfuncs;
    int sizeX5 = maxrowfuncs * maxcolsize;
    pfock->sizeX4 = sizeX4;
    pfock->sizeX5 = sizeX5;
    pfock->sizeX6 = sizeX6;
    pfock->ncpu_f = ncpu_f;
    int numF = pfock->numF = (nthreads + ncpu_f - 1)/ncpu_f;

    // allocation
    pfock->F1 = (double *)PFOCK_MALLOC(sizeof(double) * sizeX1 * numF * pfock->max_numdmat2);
    pfock->F2 = (double *)PFOCK_MALLOC(sizeof(double) * sizeX2 * numF * pfock->max_numdmat2); 
    pfock->F3 = (double *)PFOCK_MALLOC(sizeof(double) * sizeX3 *    1 * pfock->max_numdmat2);
    pfock->mem_cpu += 1.0 * sizeof(double) *
        (((double)sizeX1 + sizeX2) * numF + sizeX3) * pfock->max_numdmat2;
    if (NULL == pfock->F1 ||
        NULL == pfock->F2 ||
        NULL == pfock->F3) 
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    } 

    pfock->ldX1 = maxrowsize;
    pfock->ldX2 = maxcolsize;
    pfock->ldX3 = maxcolsize;
    pfock->ldX4 = maxcolfuncs;
    pfock->ldX5 = maxcolsize;
    pfock->ldX6 = maxcolfuncs;        
    return PFOCK_STATUS_SUCCESS;
}


static void destroy_buffers (PFock_t pfock)
{
    GTM_destroy(pfock->gtm_F1);
    GTM_destroy(pfock->gtm_F2);
    GTM_destroy(pfock->gtm_F3);
    if (pfock->getFockMatBuf != NULL) PFOCK_FREE(pfock->getFockMatBuf);
    
    PFOCK_FREE(pfock->rowpos);
    PFOCK_FREE(pfock->colpos);
    PFOCK_FREE(pfock->rowptr);
    PFOCK_FREE(pfock->colptr);
    PFOCK_FREE(pfock->loadrow);
    PFOCK_FREE(pfock->loadcol);
    PFOCK_FREE(pfock->rowsize);
    PFOCK_FREE(pfock->colsize);

    PFOCK_FREE(pfock->D_mat);
    PFOCK_FREE(pfock->F1);
    PFOCK_FREE(pfock->F2);
    PFOCK_FREE(pfock->F3);
}

static void init_mallopt()
{
    // Disable memory mapped malloc, previously done in MA_init() 
    // for caching page registrations 
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
}

PFockStatus_t PFock_create(BasisSet_t basis, int nprow, int npcol, int ntasks,
                           double tolscr, int max_numdmat, int symm,
                           PFock_t *_pfock)
{
    // Init malloc optimization
	init_mallopt();
	
    // allocate pfock
    PFock_t pfock = (PFock_t)PFOCK_MALLOC(sizeof(struct PFock));    
    if (NULL == pfock) {
        PFOCK_PRINTF(1, "Failed to allocate memory: %lld\n",
            sizeof(struct PFock));
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    memset(pfock, 0, sizeof(PFock_t));
    
    // check if MPI is initialized
    int flag;    
    MPI_Initialized(&flag);
    if (!flag) {
        PFOCK_PRINTF(1, "MPI_Init() or MPI_Init_thread()"
                     " has not been called\n");
        return PFOCK_STATUS_INIT_FAILED;        
    }
    int nprocs;
    int myrank;   
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);         
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // initialization
    pfock->nosymm = (symm == 0 ? 1 : 0);
    pfock->maxnfuncs = CInt_getMaxShellDim (basis);
    pfock->nbf = CInt_getNumFuncs (basis);
    pfock->nshells = CInt_getNumShells (basis);
    pfock->natoms = CInt_getNumAtoms (basis);
    pfock->nthreads = omp_get_max_threads ();
    pfock->mem_cpu = 0.0;
    omp_set_num_threads (pfock->nthreads);
    
    // check inputs
    if (nprow <= 0 || nprow > pfock->nshells ||
        npcol <= 0 || npcol > pfock->nshells ||
        (nprow * npcol) > nprocs) {
        PFOCK_PRINTF(1, "Invalid nprow or npcol\n");
        return PFOCK_STATUS_INVALID_VALUE;
    } else {
        pfock->nprow= nprow;
        pfock->npcol = npcol;
        pfock->nprocs = nprow * npcol;
    }
    if (tolscr < 0.0) {
        PFOCK_PRINTF(1, "Invalid screening threshold\n");
        return PFOCK_STATUS_INVALID_VALUE;
    } else {
        pfock->tolscr = tolscr;
        pfock->tolscr2 = tolscr * tolscr;
    }
    if (max_numdmat <= 0) {
        PFOCK_PRINTF(1, "Invalid number of density matrices\n");
        return PFOCK_STATUS_INVALID_VALUE;
    } else {
        pfock->max_numdmat = max_numdmat;
        pfock->max_numdmat2 = (pfock->nosymm + 1) * max_numdmat;
    }

    // set tasks
    int minnshells = (nprow > npcol ? nprow : npcol);
    minnshells = pfock->nshells/minnshells;
    if (ntasks >= minnshells) {
        pfock->nbp_p = minnshells;
    } else if (ntasks <= 0) {
        pfock->nbp_p = 4;
        pfock->nbp_p = MIN (pfock->nbp_p, minnshells);
    } else {
        pfock->nbp_p = ntasks;
    }
    pfock->nbp_row = pfock->nbp_col = pfock->nbp_p;
       
    // functions starting positions of shells
    pfock->f_startind =
        (int *)PFOCK_MALLOC(sizeof(int) * (pfock->nshells + 1));
    pfock->mem_cpu += sizeof(int) * (pfock->nshells + 1);   
    if (NULL == pfock->f_startind) {
        PFOCK_PRINTF(1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }  
    for (int i = 0; i < pfock->nshells; i++) {
        pfock->f_startind[i] = CInt_getFuncStartInd(basis, i);
    }
    pfock->f_startind[pfock->nshells] = pfock->nbf;

    // shells starting positions of atoms
    pfock->s_startind =
        (int *)PFOCK_MALLOC(sizeof(int) * (pfock->natoms + 1));
    pfock->mem_cpu += sizeof(int) * (pfock->natoms + 1); 
    if (NULL == pfock->s_startind) {
        PFOCK_PRINTF(1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    for (int i = 0; i < pfock->natoms; i++) {
        pfock->s_startind[i] = CInt_getAtomStartInd(basis, i);
    }
    pfock->s_startind[pfock->natoms] = pfock->nshells;

    PFockStatus_t ret;
    // init comm
    if ((ret = init_fock(pfock)) != PFOCK_STATUS_SUCCESS) {
        return ret;
    }
    
    // init scheduler
    if (init_taskq(pfock) != 0) {
        PFOCK_PRINTF(1, "task queue initialization failed\n");
        return PFOCK_STATUS_INIT_FAILED;
    }

    // schwartz screening    
    if (myrank == 0) {
        PFOCK_INFO("screening ...\n");
    }
    double t1 = MPI_Wtime();
    if (schwartz_screening(pfock, basis) != 0) {
        PFOCK_PRINTF (1, "schwartz screening failed\n");
        return PFOCK_STATUS_INIT_FAILED;
    }
    double t2 = MPI_Wtime();
    if (myrank == 0) {
        PFOCK_INFO("schwartz screening takes %.3lf secs\n", t2 - t1);
    }

    // repartition
    if ((ret = repartition_fock(pfock)) != PFOCK_STATUS_SUCCESS) {
        return ret;
    }

    // init global arrays
    if ((ret = create_GA(pfock)) != PFOCK_STATUS_SUCCESS) {
        return ret;
    }

    // create local buffers
    if ((ret = create_buffers(pfock)) != PFOCK_STATUS_SUCCESS) {
        return ret;
    }

    if ((ret = create_FD_GArrays(pfock)) != PFOCK_STATUS_SUCCESS) {
        return ret;
    }

    //CInt_createERD(basis, &(pfock->erd), pfock->nthreads);
    CInt_createSIMINT(basis, &(pfock->simint), pfock->nthreads);

    // statistics
    pfock->mpi_timepass
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timereduce
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timeinit
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timecomp
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timegather
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timescatter
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_usq
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_uitl
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_steals
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_stealfrom
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_ngacalls
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_volumega
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    pfock->mpi_timenexttask
        = (double *)PFOCK_MALLOC(sizeof(double) * pfock->nprocs);
    if (pfock->mpi_timepass == NULL ||
        pfock->mpi_timereduce == NULL ||
        pfock->mpi_timeinit == NULL ||
        pfock->mpi_timecomp == NULL ||        
        pfock->mpi_usq == NULL ||
        pfock->mpi_uitl == NULL ||
        pfock->mpi_steals == NULL ||
        pfock->mpi_stealfrom == NULL ||
        pfock->mpi_ngacalls == NULL ||
        pfock->mpi_volumega == NULL ||
        pfock->mpi_timegather == NULL ||
        pfock->mpi_timescatter == NULL  ||
        pfock->mpi_timenexttask == NULL) {
        PFOCK_PRINTF(1, "Mmemory allocation for statistic info failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    pfock->committed = 0;
    *_pfock = pfock;
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroy(PFock_t pfock)
{
    PFOCK_FREE(pfock->blkrowptr_sh);
    PFOCK_FREE(pfock->blkcolptr_sh);
    PFOCK_FREE(pfock->rowptr_sh);
    PFOCK_FREE(pfock->colptr_sh);
    PFOCK_FREE(pfock->rowptr_f);
    PFOCK_FREE(pfock->colptr_f);
    PFOCK_FREE(pfock->rowptr_blk);
    PFOCK_FREE(pfock->colptr_blk);
    PFOCK_FREE(pfock->FT_block);
    PFOCK_FREE(pfock->f_startind);
    PFOCK_FREE(pfock->s_startind);

    //CInt_destroyERD(pfock->erd);    
    CInt_destroySIMINT(pfock->simint, 1);    
    clean_taskq(pfock);
    clean_screening(pfock);
    destroy_GA(pfock);
    destroy_buffers(pfock);

    PFOCK_FREE(pfock->mpi_timepass);
    PFOCK_FREE(pfock->mpi_timereduce);
    PFOCK_FREE(pfock->mpi_timeinit);
    PFOCK_FREE(pfock->mpi_timecomp);
    PFOCK_FREE(pfock->mpi_timegather);
    PFOCK_FREE(pfock->mpi_timescatter);
    PFOCK_FREE(pfock->mpi_usq);
    PFOCK_FREE(pfock->mpi_uitl);
    PFOCK_FREE(pfock->mpi_steals);
    PFOCK_FREE(pfock->mpi_stealfrom);
    PFOCK_FREE(pfock->mpi_ngacalls);
    PFOCK_FREE(pfock->mpi_volumega);
    PFOCK_FREE(pfock->mpi_timenexttask);
    
    PFOCK_FREE(pfock);
  
    return PFOCK_STATUS_SUCCESS;
}

void PFock_GTM_getFockMat(
    PFock_t pfock,
    int rowstart, int rowend,
    int colstart, int colend,
    int stride,   double *mat
)
{
    int nrows = rowend - rowstart + 1;
    int ncols = colend - colstart + 1;
    
    GTM_startBatchGet(pfock->gtm_Fmat);
    GTM_addGetBlockRequest(
        pfock->gtm_Fmat,
        rowstart, nrows,
        colstart, ncols,
        mat, stride
    );
    GTM_execBatchGet(pfock->gtm_Fmat);
    GTM_stopBatchGet(pfock->gtm_Fmat);
    // Not all processes call this function, don't sync here
    //GTM_sync(pfock->gtm_Fmat);
    
    #ifndef __SCF__
    if (nrows * ncols > pfock->getFockMatBufSize)
    {
        if (pfock->getFockMatBuf != NULL) PFOCK_FREE(pfock->getFockMatBuf);
        pfock->getFockMatBufSize = pfock->getFockMatBufSize;
        pfock->getFockMatBuf     = (double*) PFOCK_MALLOC(nrows * ncols * sizeof(double));
        assert(pfock->getFockMatBuf != NULL);
    }
    double *K = pfock->getFockMatBuf;
    GTM_startBatchGet(pfock->gtm_Kmat);
    GTM_addGetBlockRequest(
        pfock->gtm_Kmat, 
        rowstart, nrows,
        colstart, ncols,
        K, ncols
    );
    GTM_execBatchGet(pfock->gtm_Kmat);
    GTM_stopBatchGet(pfock->gtm_Kmat);
    // Not all processes call this function, don't sync here
    //GTM_sync(pfock->gtm_Kmat);
    for (int i = 0; i < nrows; i++)
        #pragma vector
        for (int j = 0; j < ncols; j++)
            mat[i * stride + j] += K[i * ncols + j];
    #endif
}

PFockStatus_t PFock_computeFock(BasisSet_t basis, PFock_t pfock)
{
    struct timeval tv1;
    struct timeval tv2;
    struct timeval tv3;
    struct timeval tv4; 
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    pfock->committed = 0;      
    pfock->timepass = 0.0;
    pfock->timereduce = 0.0;
    pfock->timeinit = 0.0;
    pfock->timecomp = 0.0;
    pfock->timegather = 0.0;
    pfock->timescatter = 0.0;
    pfock->usq = 0.0;
    pfock->uitl = 0.0;
    pfock->steals = 0.0;
    pfock->stealfrom = 0.0;
    pfock->ngacalls = 0.0;
    pfock->volumega = 0.0;
    pfock->timenexttask = 0.0;
    int my_sshellrow = pfock->sshell_row;
    int my_sshellcol = pfock->sshell_col;
    int myrow = myrank/pfock->npcol;
    int mycol = myrank%pfock->npcol;
    int sizeX1 = pfock->sizeX1;
    int sizeX2 = pfock->sizeX2;
    int sizeX3 = pfock->sizeX3;
    double *F1 = pfock->F1;
    double *F2 = pfock->F2;
    double *F3 = pfock->F3;
    int maxrowsize = pfock->maxrowsize;
    int maxcolfuncs = pfock->maxcolfuncs;
    int maxcolsize = pfock->maxcolsize;
    int ldX3 = maxcolsize;
    int ldX4 = maxcolfuncs;
    int ldX5 = maxcolsize;
    int ldX6 = maxcolfuncs;
    double dzero = 0.0;
    
    init_block_buf(basis, pfock);
    
    gettimeofday (&tv1, NULL);    
    gettimeofday (&tv3, NULL);
    
    GTM_fill(pfock->gtm_Fmat, &dzero);
    GTM_fill(pfock->gtm_Kmat, &dzero);
    GTM_fill(pfock->gtm_F1, &dzero);
    GTM_fill(pfock->gtm_F2, &dzero);
    GTM_fill(pfock->gtm_F3, &dzero);
    GTM_sync(pfock->gtm_F3);
    
    // local my D
    load_full_DenMat(pfock);

    gettimeofday(&tv4, NULL);
    pfock->timegather += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    pfock->ngacalls += 3;
    pfock->volumega += (sizeX1 + sizeX2 + sizeX3) * sizeof(double);
    
    gettimeofday (&tv3, NULL);   
    reset_F(pfock->numF, pfock->num_dmat2, F1, F2, F3, sizeX1, sizeX2, sizeX3);
    gettimeofday (&tv4, NULL);
    pfock->timeinit += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    
    /* own part */
    reset_taskq(pfock);
    int task;
    int repack_D = 1;
    while ((task = taskq_next (pfock, myrow, mycol, 1)) < pfock->ntasks) 
    {
        gettimeofday (&tv3, NULL);       
        fock_task(
            pfock->nblks_col, pfock->sblk_row, pfock->sblk_col,
            task, my_sshellrow, my_sshellcol, repack_D
        );
        gettimeofday (&tv4, NULL);
        repack_D = 0;
        pfock->timecomp += (tv4.tv_sec - tv3.tv_sec) +
                    (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    } /* own part */

    gettimeofday (&tv3, NULL);     
    
    reduce_F(F1, F2, F3, maxrowsize, maxcolsize, ldX3, ldX4, ldX5, ldX6);
    
    GTM_accBlock(pfock->gtm_F1, myrank, 1, 0, sizeX1, F1, sizeX1);
    GTM_accBlock(pfock->gtm_F2, myrank, 1, 0, sizeX2, F2, sizeX2);
    GTM_accBlock(pfock->gtm_F3, myrank, 1, 0, sizeX3, F3, sizeX3);
    
    gettimeofday (&tv4, NULL);
    pfock->timereduce += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;

    // to steal
    pfock->steals = 0;
    pfock->stealfrom = 0;
    
#ifdef __DYNAMIC__
    int prevrow = myrow;
    int prevcol = mycol;   
    /* steal tasks */
    for (int idx = 0; idx < pfock->nprocs - 1; idx++) 
    {
        int vpid = (myrank + idx + 1)%pfock->nprocs;
        int vrow = vpid/pfock->npcol;
        int vcol = vpid%pfock->npcol;
        int vsblk_row  = pfock->rowptr_blk[vrow];
        int vsblk_col  = pfock->colptr_blk[vcol];
        int vnblks_col = pfock->colptr_blk[vcol + 1] - vsblk_col;
        int vsshellrow = pfock->rowptr_sh[vrow];   
        int vsshellcol = pfock->colptr_sh[vcol];
        int stealed = 0;
        int task;
        while ((task = taskq_next(pfock, vrow, vcol, 1)) < pfock->ntasks) 
        {
            gettimeofday (&tv3, NULL);
            if (0 == stealed) 
            {
                reset_F(pfock->numF, pfock->num_dmat2, F1, F2, F3, sizeX1, sizeX2, sizeX3);
  
                pfock->stealfrom++;
            }
            gettimeofday (&tv4, NULL);
            pfock->timeinit += (tv4.tv_sec - tv3.tv_sec) +
                   (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;

            gettimeofday (&tv3, NULL);
            fock_task(
                vnblks_col, vsblk_row, vsblk_col,
                task, vsshellrow, vsshellcol, 1 - stealed
            );
            gettimeofday (&tv4, NULL);
            pfock->timecomp += (tv4.tv_sec - tv3.tv_sec) +
                        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
            pfock->steals++;
            stealed = 1;
        }
        gettimeofday (&tv3, NULL);
        if (1 == stealed) 
        {
            reduce_F(F1, F2, F3, maxrowsize, maxcolsize, ldX3, ldX4, ldX5, ldX6);

            if (vrow != myrow) 
            {
                GTM_accBlock(pfock->gtm_F1, vpid, 1, 0, sizeX1, F1, sizeX1);
            } else {
                GTM_accBlock(pfock->gtm_F1, myrank, 1, 0, sizeX1, F1, sizeX1);
            }
            
            if (vcol != mycol) 
            {
                GTM_accBlock(pfock->gtm_F2, vpid, 1, 0, sizeX2, F2, sizeX2);
            } else {
                GTM_accBlock(pfock->gtm_F2, myrank, 1, 0, sizeX2, F2, sizeX2);
            }
            
            GTM_accBlock(pfock->gtm_F3, vpid, 1, 0, sizeX3, F3, sizeX3);
            prevrow = vrow;
            prevcol = vcol;
        }
        gettimeofday (&tv4, NULL);
        pfock->timereduce += (tv4.tv_sec - tv3.tv_sec) +
                        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    } /* steal tasks */    
#endif /* #ifdef __DYNAMIC__ */

    GTM_sync(pfock->gtm_F3);
    
    gettimeofday (&tv2, NULL);
    pfock->timepass = (tv2.tv_sec - tv1.tv_sec) +
               (tv2.tv_usec - tv1.tv_usec) / 1000.0 / 1000.0;    
    
    gettimeofday (&tv3, NULL);
    store_local_bufF (pfock);
    gettimeofday (&tv4, NULL);
    pfock->timescatter = (tv4.tv_sec - tv3.tv_sec) +
               (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;

    if (myrank == 0) {
        PFOCK_INFO ("correct F ...\n");
    }
  
    if (pfock->nosymm) 
    {
        // GTMatrix cannot handle this yet...
        /*
        double dhalf = 0.5;
        for (int i = 0; i < pfock->num_dmat; i++) {
            GA_Transpose(pfock->ga_F[i + pfock->num_dmat],
                         pfock->ga_D[0]);
            GA_Add(&dhalf, pfock->ga_F[i],
                   &dhalf, pfock->ga_D[0], pfock->ga_F[i]);
        #ifndef __SCF__
            GA_Transpose(pfock->ga_K[i + pfock->num_dmat],
                         pfock->ga_D[0]);
            GA_Add(&dhalf, pfock->ga_K[i],
                   &dhalf, pfock->ga_D[0], pfock->ga_K[i]);
        #endif
        }
        */
    } else {
        // correct F
        GTM_symmetrize(pfock->gtm_Fmat);
        #ifndef __SCF__
        GTM_symmetrize(pfock->gtm_Kmat);
        #endif
    }
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_createCoreHMat(PFock_t pfock, BasisSet_t basis)
{   
    int stride;
    double *mat, dzero = 0.0;

    GTM_fill(pfock->gtm_Hmat, &dzero);
    mat    = pfock->gtm_Hmat->mat_block;
    stride = pfock->gtm_Hmat->ld_local;
    compute_H(pfock, basis, pfock->sshell_row, pfock->eshell_row,
              pfock->sshell_col, pfock->eshell_col, stride, mat);
    GTM_sync(pfock->gtm_Hmat);
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroyCoreHMat(PFock_t pfock)
{
    GTM_destroy(pfock->gtm_Hmat);
    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getCoreHMat(PFock_t pfock, int rowstart, int rowend,
                                int colstart, int colend,
                                int stride, double *mat)
{
    GTM_startBatchGet(pfock->gtm_Hmat);
    GTM_addGetBlockRequest(
        pfock->gtm_Hmat, 
        rowstart, rowend - rowstart + 1,
        colstart, colend - colstart + 1,
        mat, stride
    );
    GTM_execBatchGet(pfock->gtm_Hmat);
    GTM_stopBatchGet(pfock->gtm_Hmat);
    // Not all processes call this function, don't sync here
    //GTM_sync(pfock->gtm_Hmat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_createOvlMat(PFock_t pfock, BasisSet_t basis)
{
    double *mat;
    int stride;
    double dzero = 0.0;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // (1) Compute S
    GTM_fill(pfock->gtm_Smat, &dzero);
    mat    = pfock->gtm_Smat->mat_block;
    stride = pfock->gtm_Smat->ld_local;
    compute_S(pfock, basis, pfock->sshell_row, pfock->eshell_row,
              pfock->sshell_col, pfock->eshell_col, stride, mat);
    GTM_sync(pfock->gtm_Smat);
    
    // (2) Compute X         
    int nbf = CInt_getNumFuncs(basis);
    double *eval = (double *)malloc(nbf * sizeof (double));
    if (NULL == eval) 
    {
        PFOCK_PRINTF (1, "Memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;        
    }
    
    my_peig(pfock->gtm_Smat, pfock->gtm_tmp1, nbf, pfock->nprow, pfock->npcol, eval);
    
    GTMatrix_t gtm_tmp1 = pfock->gtm_tmp1;
    GTMatrix_t gtm_tmp2 = pfock->gtm_tmp2;
    double *blocktmp = gtm_tmp1->mat_block;
    double *blockS   = gtm_tmp2->mat_block;
    int nfuncs_row   = gtm_tmp1->r_blklens[gtm_tmp1->my_rowblk];
    int nfuncs_col   = gtm_tmp1->c_blklens[gtm_tmp1->my_colblk];
    int ld     = gtm_tmp1->ld_local;
    int lo1tmp = gtm_tmp1->c_displs[gtm_tmp1->my_colblk];

    double *lambda_vector = (double *)malloc(nfuncs_col * sizeof (double));
    assert (lambda_vector != NULL);   

    #pragma simd
    for (int j = 0; j < nfuncs_col; j++) 
        lambda_vector[j] = 1.0 / sqrt(eval[j + lo1tmp]);
    free(eval);
    
    for (int i = 0; i < nfuncs_row; i++)  
    {
        #pragma simd
        for (int j = 0; j < nfuncs_col; j++) 
            blockS[i * ld + j] = blocktmp[i * ld + j] * lambda_vector[j];
    }
    free(lambda_vector);

    double t1 = MPI_Wtime();
    
    GTMatrix_t gtm_Xmat = pfock->gtm_Xmat;
    int nrows_X = gtm_Xmat->r_blklens[gtm_Xmat->my_rowblk];
    int ncols_X = gtm_Xmat->c_blklens[gtm_Xmat->my_colblk];
    int X_row_s = gtm_Xmat->r_displs[gtm_Xmat->my_rowblk];
    int X_col_s = gtm_Xmat->c_displs[gtm_Xmat->my_colblk];
    double *tmp1_buf = (double*) _mm_malloc(nrows_X * nbf * sizeof(double), 64);
    double *tmp2_buf = (double*) _mm_malloc(nbf * ncols_X * sizeof(double), 64);

    GTM_startBatchGet(gtm_tmp1);
    GTM_addGetBlockRequest(gtm_tmp1, X_row_s, nrows_X, 0, nbf, tmp1_buf, nbf);
    GTM_execBatchGet(gtm_tmp1);
    GTM_stopBatchGet(gtm_tmp1);
    GTM_sync(gtm_tmp1);

    GTM_startBatchGet(gtm_tmp2);
    GTM_addGetBlockRequest(gtm_tmp2, X_col_s, ncols_X, 0, nbf, tmp2_buf, nbf);
    GTM_execBatchGet(gtm_tmp2);
    GTM_stopBatchGet(gtm_tmp2);
    GTM_sync(gtm_tmp2);

    double *gtm_X_block = gtm_Xmat->mat_block;
    int gtm_X_ld = gtm_Xmat->ld_local;

    GTM_fill(gtm_Xmat, &dzero);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        nrows_X, ncols_X, nbf,
        1.0,
        tmp1_buf, nbf,
        tmp2_buf, nbf,
        0.0,
        gtm_X_block, gtm_X_ld
    );

    _mm_free(tmp1_buf);
    _mm_free(tmp2_buf);
    
    GTM_sync(pfock->gtm_Xmat);

    double t2 = MPI_Wtime() - t1;
    double tmax;
    MPI_Reduce(&t2, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0) printf("  My PDGEMM used time = %lf (s)\n", tmax);
    
    GTM_destroy(pfock->gtm_tmp1);
    GTM_destroy(pfock->gtm_tmp2);

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroyOvlMat(PFock_t pfock)
{
    GTM_destroy(pfock->gtm_Xmat);
    GTM_destroy(pfock->gtm_Smat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getOvlMat(PFock_t pfock, int rowstart, int rowend,
                              int colstart, int colend,
                              int stride, double *mat)
{
    GTM_startBatchGet(pfock->gtm_Smat);
    GTM_addGetBlockRequest(
        pfock->gtm_Smat, 
        rowstart, rowend - rowstart + 1,
        colstart, colend - colstart + 1,
        mat, stride
    );
    GTM_execBatchGet(pfock->gtm_Smat);
    GTM_stopBatchGet(pfock->gtm_Smat);
    // Not all processes call this function, don't sync here
    //GTM_sync(pfock->gtm_Smat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getOvlMat2(PFock_t pfock, int rowstart, int rowend,
                               int colstart, int colend,
                               int stride, double *mat)
{
    GTM_startBatchGet(pfock->gtm_Xmat);
    GTM_addGetBlockRequest(
        pfock->gtm_Xmat, 
        rowstart, rowend - rowstart + 1,
        colstart, colend - colstart + 1,
        mat, stride
    );
    GTM_execBatchGet(pfock->gtm_Xmat);
    GTM_stopBatchGet(pfock->gtm_Xmat);
    // Not all processes call this function, don't sync here
    //GTM_sync(pfock->gtm_Xmat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getMemorySize(PFock_t pfock, double *mem_cpu)
{
    *mem_cpu = pfock->mem_cpu;  
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getStatistics(PFock_t pfock)
{
    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    
    // statistics
    MPI_Gather (&pfock->steals, 1, MPI_DOUBLE,
        pfock->mpi_steals, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->stealfrom, 1, MPI_DOUBLE,
        pfock->mpi_stealfrom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->usq, 1, MPI_DOUBLE, 
        pfock->mpi_usq, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->uitl, 1, MPI_DOUBLE, 
        pfock->mpi_uitl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timepass, 1, MPI_DOUBLE, 
        pfock->mpi_timepass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timecomp, 1, MPI_DOUBLE, 
        pfock->mpi_timecomp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timeinit, 1, MPI_DOUBLE, 
        pfock->mpi_timeinit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timereduce, 1, MPI_DOUBLE, 
        pfock->mpi_timereduce, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timegather, 1, MPI_DOUBLE, 
        pfock->mpi_timegather, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timescatter, 1, MPI_DOUBLE, 
        pfock->mpi_timescatter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->volumega, 1, MPI_DOUBLE, 
        pfock->mpi_volumega, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->ngacalls, 1, MPI_DOUBLE, 
        pfock->mpi_ngacalls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timenexttask, 1, MPI_DOUBLE, 
        pfock->mpi_timenexttask, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        double total_timepass;
        double max_timepass;
        double total_timereduce;
        double total_timeinit;
        double total_timecomp;
        double total_timegather;
        double total_timescatter;
        double total_usq;
        double max_usq;
        double total_uitl;
        double max_uitl;
        double total_steals;
        double total_stealfrom;
        double total_ngacalls;
        double total_volumega;
        double total_timenexttask;
        for (int i = 0; i < pfock->nprocs; i++) {
            total_timepass += pfock->mpi_timepass[i];
            max_timepass =
                max_timepass < pfock->mpi_timepass[i] ?
                    pfock->mpi_timepass[i] : max_timepass;            
            total_usq += pfock->mpi_usq[i];
            max_usq = 
                max_usq < pfock->mpi_usq[i] ? pfock->mpi_usq[i] : max_usq;          
            total_uitl += pfock->mpi_uitl[i];
            max_uitl = 
                max_uitl < pfock->mpi_uitl[i] ? pfock->mpi_uitl[i] : max_uitl;           
            total_steals += pfock->mpi_steals[i];
            total_stealfrom += pfock->mpi_stealfrom[i];
            total_timecomp += pfock->mpi_timecomp[i];
            total_timeinit += pfock->mpi_timeinit[i];
            total_timereduce += pfock->mpi_timereduce[i];
            total_timegather += pfock->mpi_timegather[i];
            total_timescatter += pfock->mpi_timescatter[i];
            total_ngacalls += pfock->mpi_ngacalls[i];
            total_volumega += pfock->mpi_volumega[i];
            total_timenexttask += pfock->mpi_timenexttask[i];
        }
        double tsq = pfock->nshells;
        tsq = ((tsq + 1) * tsq/2.0 + 1) * tsq * (tsq + 1)/4.0;
        printf("    PFock Statistics:\n");
        printf("      average totaltime   = %.3g\n"
               "      average timegather  = %.3g\n"
               "      average timeinit    = %.3g\n"
               "      average timecomp    = %.3g\n"
               "      average timereduce  = %.3g\n"
               "      average timenexttask= %.3g\n"
               "      average timescatter = %.3g\n"
               "      comp/total = %.3g\n",
               total_timepass/pfock->nprocs,
               total_timegather/pfock->nprocs,
               total_timeinit/pfock->nprocs,
               total_timecomp/pfock->nprocs,
               total_timereduce/pfock->nprocs,
               total_timenexttask/pfock->nprocs,
               total_timescatter/pfock->nprocs,
               total_timecomp/total_timepass);
        printf("      usq = %.4g (lb = %.3g)\n"
               "      uitl = %.4g (lb = %.3g)\n"
               "      nsq = %.4g (screening = %.3g)\n",
               total_usq, max_usq/(total_usq/pfock->nprocs),
               total_uitl, max_uitl/(total_uitl/pfock->nprocs),
               tsq, total_usq/tsq);
        printf("      load blance = %.3lf\n",
               max_timepass/(total_timepass/pfock->nprocs));
        printf("      steals = %.3g (average = %.3g)\n"
               "      stealfrom = %.3g (average = %.3g)\n"
               "      GAcalls = %.3g\n"
               "      GAvolume %.3g MB\n",
               total_steals, total_steals/pfock->nprocs,
               total_stealfrom, total_stealfrom/pfock->nprocs,
               total_ngacalls/pfock->nprocs,
               total_volumega/pfock->nprocs/1024.0/1024.0);
    }
    
    return PFOCK_STATUS_SUCCESS;
}
