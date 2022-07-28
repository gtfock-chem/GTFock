#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
//#include <ga.h>
#include <mpi.h>

#include "CInt.h"
#include "config.h"
#include "screening.h"
#include "taskq.h"

#include "cint_basisset.h"

#include "GTMatrix.h"

/* Note on 4-index permutation to rearrange output from integral library:

If original code accesses an integral as:

  integrals[iM + dimM*(iN + dimN * (iP + dimP * iQ))]

it should be changed to:

  integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];//Simint 

If the original code accesses an integral as:
  iM * (dimN*dimM*dimN + dimN) + iN * (dimM*dimN+1)

which corresponds to:
  iN + dimN*(iM + dimM * (iN + dimN * iM))

This should be changed to:
  iM + dimM*(iN + dimN * (iM + dimM * iN))

which corresponds to:
  iM * (dimM*dimN+1) + iN * (dimM + dimM*dimN*dimM) //Simint
*/

static int cmp_pair(int M1, int N1, int M2, int N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

static void quickSort(int *M, int *N, double *shell_val, int l, int r)
{
    int i = l, j = r, tmp;
    int mid_M = M[(i + j) / 2];
    int mid_N = N[(i + j) / 2];
    double dtmp;
    while (i <= j)
    {
        while (cmp_pair(M[i], N[i], mid_M, mid_N)) i++;
        while (cmp_pair(mid_M, mid_N, M[j], N[j])) j--;
        if (i <= j)
        {
            tmp = M[i]; M[i] = M[j]; M[j] = tmp;
            tmp = N[i]; N[i] = N[j]; N[j] = tmp;
            
            dtmp = shell_val[i];
            shell_val[i] = shell_val[j];
            shell_val[j] = dtmp;
            
            i++;  j--;
        }
    }
    if (i < r) quickSort(M, N, shell_val, i, r);
    if (j > l) quickSort(M, N, shell_val, l, j);
}

int schwartz_screening(PFock_t pfock, BasisSet_t basis)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 

    // create shell pairs values    
    SIMINT_t simint;
    int nthreads = omp_get_max_threads();
    CInt_createSIMINT(basis, &simint, nthreads);  
    
    // create global arrays for screening 
    int nprow = pfock->nprow;
    int npcol = pfock->npcol;
    int nshells = pfock->nshells;
    GTM_create(
        &pfock->gtm_scrval, MPI_COMM_WORLD, MPI_DOUBLE, 8,
        myrank, nshells, nshells, nprow, npcol,
        pfock->rowptr_sh, pfock->colptr_sh
    );

    // compute the max shell value
    int num_sq_values = pfock->nshells_row * pfock->nshells_col;
    double *sq_values = (double *)PFOCK_MALLOC(sizeof(double) * num_sq_values);
    if (NULL == sq_values) return -1;
    
    int startM = pfock->sshell_row;
    int startN = pfock->sshell_col;
    int endM = pfock->eshell_row;
    int endN = pfock->eshell_col;
    double maxtmp = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for reduction(max:maxtmp)
        for (int M = startM; M <= endM; M++) 
        {
            int dimM = CInt_getShellDim(basis, M);
            for (int N = startN; N <= endN; N++) 
            {
                int dimN = CInt_getShellDim(basis, N);
                int nints;
                double *integrals;
                CInt_computeShellQuartet_SIMINT(simint, tid, M, N, M, N, &integrals, &nints);            
                double maxvalue = 0.0;
                if (nints != 0) 
                {
                    for (int iM = 0; iM < dimM; iM++) 
                    {
                        for (int iN = 0; iN < dimN; iN++) 
                        {
                            int index = 
                                iN * (dimM*dimN*dimM+dimM) + iM * (dimN*dimM+1);//Simint
                              //iM * (dimN*dimM*dimN+dimN) + iN * (dimM*dimN+1);//OptERD
                            if (maxvalue < fabs(integrals[index]))
                                maxvalue = fabs(integrals[index]); 
                        }
                    }
                }
                sq_values[(M - startM) * (endN - startN + 1)  + (N - startN)] = maxvalue;
                if (maxvalue > maxtmp) maxtmp = maxvalue;
            }
        }
    }
    int lo[2] = {startM, startN};
    int hi[2] = {endM, endN};
    int ld = endN - startN + 1;
    GTM_startBatchPut(pfock->gtm_scrval);
    GTM_addPutBlockRequest(
        pfock->gtm_scrval, 
        lo[0], hi[0] - lo[0] + 1,
        lo[1], hi[1] - lo[1] + 1,
        sq_values, ld
    );
    GTM_execBatchPut(pfock->gtm_scrval);
    GTM_stopBatchPut(pfock->gtm_scrval);
    GTM_sync(pfock->gtm_scrval);
    
    // max value
    MPI_Allreduce(&maxtmp, &(pfock->maxvalue), 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    PFOCK_FREE(sq_values);

    // init shellptr
    sq_values = (double *)PFOCK_MALLOC(sizeof(double) * nshells * nshells);
    if (NULL == sq_values) return -1;
    
    int nnz = 0;
    double eta = pfock->tolscr2 / pfock->maxvalue;
    pfock->shellptr = (int *)PFOCK_MALLOC(sizeof(int) * (nshells + 1));
    pfock->mem_cpu += 1.0 * sizeof(int) * (nshells + 1);
    if (NULL == pfock->shellptr) return -1;
    memset(pfock->shellptr, 0, sizeof(int) * (nshells + 1));
    
    GTM_startBatchGet(pfock->gtm_scrval);
    GTM_addGetBlockRequest(pfock->gtm_scrval, 0, nshells, 0, nshells, sq_values, nshells);
    GTM_execBatchGet(pfock->gtm_scrval);
    GTM_stopBatchGet(pfock->gtm_scrval);
    GTM_sync(pfock->gtm_scrval);
    
    for (int M = 0; M < nshells; M++) 
    {
        pfock->shellptr[M] = nnz;
        double *sq_values_M = sq_values + M * nshells;
        for (int N = 0; N < nshells; N++) 
        {
            double maxvalue = sq_values_M[N];
            if (maxvalue > eta) 
            {
                if (M > N && (M + N) % 2 == 1 || M < N && (M + N) % 2 == 0)  continue;
                else nnz++;
            }
        }
        pfock->shellptr[M + 1] = nnz;
    }
    pfock->nnz = nnz;
    
    double maxvalue;  
    pfock->shellvalue = (double *) PFOCK_MALLOC(sizeof(double) * nnz);
    pfock->shellid    = (int *)    PFOCK_MALLOC(sizeof(int)    * nnz);
    pfock->shellrid   = (int *)    PFOCK_MALLOC(sizeof(int)    * nnz);
    pfock->mem_cpu += 1.0 * sizeof(double) * nnz + 2.0 * sizeof(int) * nnz;
    nshells = pfock->nshells;
    if (pfock->shellvalue == NULL ||
        pfock->shellid == NULL ||
        pfock->shellrid == NULL) {
        return -1;
    }    
    
    // Check environment variables to see if we need to swap
    // shell pairs according to their angular momentum
    char *swap_by_AM_str = getenv("SWAP_BY_AM");
    int swap_by_AM = 1;
    if (swap_by_AM_str != NULL)
    {
        swap_by_AM = atoi(swap_by_AM_str);
        if ((swap_by_AM != 0) && (swap_by_AM != 1)) swap_by_AM = 1;
    }
    if (myrank == 0)
    {
        if (swap_by_AM) printf("  SWAP_BY_AM enabled\n");
        else printf("  SWAP_BY_AM disabled\n");
    }
    
    nnz = 0;
    if (swap_by_AM)
    {
        // Swap (AB) to (BA) if AM(B) > AM(A)
        for (int A = 0; A < nshells; A++) 
        {
            double *sq_values_A = sq_values + A * nshells;
            for (int B = 0; B < nshells; B++) 
            {
                maxvalue = sq_values_A[B];
                if (maxvalue > eta) 
                {
                    if (A > B && (A + B) % 2 == 1 || A < B && (A + B) % 2 == 0) continue;
                    
                    // Don't need to change the shellvalue, since it is the same for (AB) and (BA)
                    if (A == B) 
                    {
                        pfock->shellvalue[nnz] =  maxvalue;                       
                    } else {
                        pfock->shellvalue[nnz] = -maxvalue;
                    }
                    
                    int AB_id = CInt_SIMINT_getShellpairAMIndex(simint, A, B);
                    int BA_id = CInt_SIMINT_getShellpairAMIndex(simint, B, A);
                    if (AB_id > BA_id)
                    {
                        pfock->shellrid[nnz] = A;    
                        pfock->shellid[nnz]  = B;
                    } else {
                        pfock->shellrid[nnz] = B;    
                        pfock->shellid[nnz]  = A;
                    }
                    
                    nnz++;
                }
            }
        }
    } else {
        for (int A = 0; A < nshells; A++) 
        {
            pfock->shellptr[A] = nnz;
            double *sq_values_A = sq_values + A * nshells;
            for (int B = 0; B < nshells; B++) 
            {
                maxvalue = sq_values_A[B];
                if (maxvalue > eta) 
                {
                    if (A > B && (A + B) % 2 == 1 || A < B && (A + B) % 2 == 0) continue;
                    if (A == B) {
                        pfock->shellvalue[nnz] = maxvalue;                       
                    } else {
                        pfock->shellvalue[nnz] = -maxvalue;
                    }
                    pfock->shellid[nnz] = B;
                    pfock->shellrid[nnz] = A;
                    nnz++;
                }
            }
        }
    }

    if (swap_by_AM)
    {
        // Must sort the shell pairs and reconstruct pfock->shellptr
        quickSort(pfock->shellrid, pfock->shellid, pfock->shellvalue, 0, nnz - 1);
        
        int shell_id = 0;
        pfock->shellptr[shell_id] = 0;
        shell_id++;
        for (int i = 1; i < nnz; i++)
        {
            if (pfock->shellrid[i] != pfock->shellrid[i - 1])
            {
                pfock->shellptr[shell_id] = i;
                shell_id++;
            }
        }
    }
    
    PFOCK_FREE(sq_values);
    CInt_destroySIMINT(simint, 0);
    GTM_destroy(pfock->gtm_scrval);
    
    return 0;
}


void clean_screening(PFock_t pfock)
{
    PFOCK_FREE(pfock->shellid);
    PFOCK_FREE(pfock->shellrid);
    PFOCK_FREE(pfock->shellptr);
    PFOCK_FREE(pfock->shellvalue);
}
