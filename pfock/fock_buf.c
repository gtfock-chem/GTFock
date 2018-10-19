#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <ga.h>
#include <macdecls.h>
#include <sys/time.h>

#include "config.h"
#include "taskq.h"
#include "fock_buf.h"

#include "Buzz_Matrix.h"

// Load full density matrix, num_dmat2 should be 1
void load_full_DenMat(PFock_t pfock)
{
    Buzz_startBatchGet(pfock->bm_Dmat);
    Buzz_addGetBlockRequest(pfock->bm_Dmat, 0, pfock->nbf, 0, pfock->nbf, pfock->D_mat, pfock->nbf);
    Buzz_execBatchGet(pfock->bm_Dmat);
    Buzz_stopBatchGet(pfock->bm_Dmat);
    Buzz_Sync(pfock->bm_Dmat);
}

void store_local_bufF(PFock_t pfock)
{
    int *loadrow = pfock->loadrow;
    int *loadcol = pfock->loadcol;
    int sizerow = pfock->sizeloadrow;
    int sizecol = pfock->sizeloadcol;
    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    int lo[2];
    int hi[2];
    int ldF;
    int *ga_J = pfock->ga_F;
#ifdef __SCF__
    int *ga_K = pfock->ga_F;
#else
    int *ga_K = pfock->ga_K;
#endif
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    for (int i = 0; i < pfock->num_dmat2; i++) {
    #ifdef GA_NB    
        ga_nbhdl_t nbnb;
    #endif
        // local buffers
        double *F1;
        double *F2;
        double *F3;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Access(pfock->ga_F1[i], lo, hi, &F1, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access(pfock->ga_F2[i], lo, hi, &F2, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access(pfock->ga_F3[i], lo, hi, &F3, &ldF);
        int ldF1 = pfock->ldX1;
        int ldF2 = pfock->ldX2;
        int ldF3 = pfock->ldX3;    
        // update F1
        double done = 1.0;
        lo[0] = pfock->sfunc_row;
        hi[0] = pfock->efunc_row;
        for (int A = 0; A < sizerow; A++) {
            lo[1] = loadrow[PLEN * A + P_LO];
            hi[1] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
        #ifdef GA_NB
            NGA_NbAcc(ga_J[i], lo, hi, &(F1[posrow]),
                      &ldF1, &done, &nbnb);    
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F1[posrow]), &ldF1, &done);
        #endif
        }

        // update F2
        lo[0] = pfock->sfunc_col;
        hi[0] = pfock->efunc_col;
        for (int B = 0; B < sizecol; B++) {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            int poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB
            NGA_NbAcc(ga_J[i], lo, hi, &(F2[poscol]),
                      &ldF2, &done, &nbnb);
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F2[poscol]), &ldF2, &done);
        #endif
        }

        // update F3
        for (int A = 0; A < sizerow; A++) {
            lo[0] = loadrow[PLEN * A + P_LO];
            hi[0] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
            for (int B = 0; B < sizecol; B++) {
                lo[1] = loadcol[PLEN * B + P_LO];
                hi[1] = loadcol[PLEN * B + P_HI];
                int poscol = loadcol[PLEN * B + P_W];
            #ifdef GA_NB
                NGA_NbAcc(ga_K[i], lo, hi, 
                          &(F3[posrow * ldF3 + poscol]), &ldF3, &done, &nbnb);
            #else
                NGA_Acc(ga_K[i], lo, hi, 
                        &(F3[posrow * ldF3 + poscol]), &ldF3, &done);        
            #endif
            }
        }
    #ifdef GA_NB
        NGA_NbWait(&nbnb);
    #endif
        // update release
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Release(pfock->ga_F1[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Release(pfock->ga_F2[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Release(pfock->ga_F3[i], lo, hi);
    }
    GA_Sync();
}


void compute_FD_ptr(PFock_t pfock, int startM, int endM, int *ptrrow, int *rowsize)
{
    for (int A = 0; A < pfock->nshells; A++) {
        ptrrow[A] = -1;
    }    
    // init row pointers
    for (int A = startM; A <= endM; A++) {
        int start = pfock->shellptr[A];
        int end = pfock->shellptr[A + 1]; 
        for (int i = start; i < end; i++) {
            int B = pfock->shellid[i];
            ptrrow[B] = 1;
        }
    }
    for (int i = 0; i < pfock->natoms; i++)
    {
        int start = pfock->s_startind[i];
        int end = pfock->s_startind[i + 1];
        int flag = -1;
        for (int A = start; A < end; A++)
        {
            if (ptrrow[A] != -1)
                flag = 1;
        }
        for (int A = start; A < end; A++)
        {
            ptrrow[A] = flag;
        }
    }
    *rowsize = 0;
    for (int A = 0; A < pfock->nshells; A++)
    {
        if (ptrrow[A] == 1)
        {
            ptrrow[A] = *rowsize;           
            *rowsize += pfock->f_startind[A + 1] - pfock->f_startind[A];
        }
    }
}


void init_FD_load(PFock_t pfock, int *ptrrow, int **loadrow, int *loadsize)
{    
    int loadcount = 0;
    for (int A = 0; A < pfock->nshells; A++) {
        if (ptrrow[A] != -1) {
            while (A < pfock->nshells && ptrrow[A] != -1) {
                A++;
            }           
            loadcount++;
        }
    }
    *loadrow = (int *)PFOCK_MALLOC(sizeof(int) * PLEN * loadcount);
    assert(NULL != *loadrow);
    *loadsize = loadcount;
    
    loadcount = 0;
    for (int A = 0; A < pfock->nshells; A++) {
        int idx = ptrrow[A];
        if (idx != -1) {
            int lo = pfock->f_startind[A];
            while (A < pfock->nshells && ptrrow[A] != -1) {
                A++;
            }           
            int hi = pfock->f_startind[A] - 1;
            (*loadrow)[loadcount * PLEN + P_LO] = lo;
            (*loadrow)[loadcount * PLEN + P_HI] = hi;
            (*loadrow)[loadcount * PLEN + P_W] = idx;
            loadcount++;
        }
    }
}
