#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <macdecls.h>
#include <sys/time.h>

#include "config.h"
#include "taskq.h"
#include "fock_task.h"
#include "cint_basisset.h"

double *update_F_buf  = NULL;
int update_F_buf_size = 0;
int use_atomic_add    = 1;

int nbf = 0, nshells = 0, nsp, nbf2;
int *mat_block_ptr;
volatile int *block_packed;
int *shell_bf_num;
double *D_blocks;

#include "update_F.h"

#define UPDATE_F_OPT_BUFFER_ARGS \
    tid, num_dmat, &batch_integrals[ipair * batch_nints], \
    fock_info_list[0],  \
    fock_info_list[1],  \
    fock_info_list[2],  \
    fock_info_list[3],  \
    fock_info_list[4],  \
    fock_info_list[5],  \
    fock_info_list[6],  \
    fock_info_list[7],  \
    fock_info_list[8],  \
    fock_info_list[9],  \
    fock_info_list[10], \
    fock_info_list[11], \
    fock_info_list[12], \
    fock_info_list[13], \
    fock_info_list[14], \
    fock_info_list[15], \
    D1, D2, D3, \
    F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP, \
    sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6, \
    ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, \
    load_MN, load_P, write_MN, write_P, \
    M, N, P_list[ipair], Q_list[ipair]

#include "thread_quartet_buf.h"

void update_F_with_KetShellPairList(
    int tid, int num_dmat, double *batch_integrals, int batch_nints, int npairs, 
    KetShellPairList_s *target_shellpair_list,
    double **D1, double **D2, double **D3,
    double *F_MN, double *F_PQ, double *F_NQ, double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3, int sizeX4, int sizeX5, int sizeX6,
    int ldX1, int ldX2, int ldX3, int ldX4, int ldX5, int ldX6, int M, int N
)
{
    int load_MN, load_P, write_MN, write_P;
    int prev_iMP = -1;
    int *P_list = target_shellpair_list->P_list;
    int *Q_list = target_shellpair_list->Q_list;
    for (int ipair = 0; ipair < npairs; ipair++)
    {
        int *fock_info_list = target_shellpair_list->fock_quartet_info + ipair * 16;

        if (ipair == 0)          load_MN  = 1; else load_MN  = 0;
        if (ipair + 1 == npairs) write_MN = 1; else write_MN = 0;
        
        if (prev_iMP == fock_info_list[9])  // iMP == previous iMP, P == previous P
        {
            load_P = 0;
        } else {
            load_P = 1;
        }
        
        write_P = 0;
        if (ipair + 1 == npairs) write_P = 1;
        if (ipair < npairs - 1)
        {
            // fock_info_list[25] == the fock_info_list[9] for ipair+1
            if (fock_info_list[9] != fock_info_list[25]) write_P = 1;
        }
        
        prev_iMP = fock_info_list[9];
        
        // Just use a lazy way here. Actually we can still do the same thing when
        // num_dmat > 1, but it needs to inline update_F to this function. 
        if (num_dmat > 1)  
        {
            load_MN  = 1;
            load_P   = 1;
            write_MN = 1;
            write_P  = 1;
        }

        int is_1111 = fock_info_list[0] * fock_info_list[1] * fock_info_list[2] * fock_info_list[3];
        if (is_1111 == 1)
        {
            update_F_1111(UPDATE_F_OPT_BUFFER_ARGS);
        } else {
            if (fock_info_list[3] == 1) update_F_opt_buffer_Q1(UPDATE_F_OPT_BUFFER_ARGS);
            else if (fock_info_list[3] == 3)  update_F_opt_buffer_Q3(UPDATE_F_OPT_BUFFER_ARGS);
            else if (fock_info_list[3] == 6)  update_F_opt_buffer_Q6(UPDATE_F_OPT_BUFFER_ARGS);
            else if (fock_info_list[3] == 10) update_F_opt_buffer_Q10(UPDATE_F_OPT_BUFFER_ARGS);
            else if (fock_info_list[3] == 15) update_F_opt_buffer_Q15(UPDATE_F_OPT_BUFFER_ARGS);
            else update_F_opt_buffer(UPDATE_F_OPT_BUFFER_ARGS);
        }
    }
}

static void init_block_buf(int _nbf, int _nshells, int *f_startind, int num_dmat)
{
    if (num_dmat != 1)
    {
        printf("  FATAL: currently JKD blocking only supports num_dmat==1 !! Please check scf.c\n");
        assert(num_dmat == 1);
    }
    
    if (nbf > 0)
    {
        if ((nbf != _nbf) || (nshells != _nshells)) 
        {
            printf("Old nbf, nshells != new nbf, nshells !!\n");
            assert(nbf == _nbf);
            assert(nshells == _nshells);
        }
        return;
    }
    
    nbf     = _nbf;
    nshells = _nshells;
    nsp     = nshells * nshells;
    nbf2    = nbf * nbf;
    
    shell_bf_num  = (int*)          malloc(sizeof(int) * nshells);
    mat_block_ptr = (int*)          malloc(sizeof(int) * nsp);
    block_packed  = (volatile int*) malloc(sizeof(int) * nsp);
    D_blocks      = (double*)       malloc(sizeof(double) * nbf2);
    assert(mat_block_ptr != NULL);
    assert(block_packed  != NULL);
    assert(shell_bf_num  != NULL);
    assert(D_blocks      != NULL);
    
    for (int i = 0; i < nshells; i++)
        shell_bf_num[i] = f_startind[i + 1] - f_startind[i];
    
    int pos = 0, idx = 0;
    for (int i = 0; i < nshells; i++)
    {
        for (int j = 0; j < nshells; j++)
        {
            mat_block_ptr[idx] = pos;
            pos += shell_bf_num[i] * shell_bf_num[j];
            idx++;
        }
    }
}

static inline void copy_matrix_block(
    double *dst, const int ldd, const double *src, const int lds, 
    const int nrows, const int ncols
)
{
    for (int irow = 0; irow < nrows; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, sizeof(double) * ncols);
}

static inline void pack_D_block(
    const int M, const int N, const int dimM, const int dimN,
    double *D_src, const int ld_Dsrc
)
{
    int MN_id = M * nshells + N;
    if (block_packed[MN_id] == 0)
    {
        double *D_dst = D_blocks + mat_block_ptr[MN_id];
        copy_matrix_block(D_dst, dimN, D_src, ld_Dsrc, dimM, dimN);
        block_packed[MN_id] = 1;
    }
}

// for SCF, J = K
// Batched ERI version
void fock_task(
    BasisSet_t basis, SIMINT_t simint, int ncpu_f, int num_dmat,
    int *shellptr, double *shellvalue,
    int *shellid, int *shellrid, int *f_startind,
    int *rowpos, int *colpos, int *rowptr, int *colptr,
    double tolscr2, int startrow, int startcol,
    int startM, int endM, int startP, int endP,
    double **D1, double **D2, double **D3,
    double *F1, double *F2, double *F3,
    double *F4, double *F5, double *F6,
    int ldX1, int ldX2, int ldX3,
    int ldX4, int ldX5, int ldX6,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    double *nitl, double *nsq, int _nbf, int _nshells
)
{
    int startMN = shellptr[startM];
    int endMN = shellptr[endM + 1];
    int startPQ = shellptr[startP];
    int endPQ = shellptr[endP + 1];
    
    if (update_F_buf_size == 0)
    {
        int maxAM, max_buf_entry_size;
        _maxMomentum(basis, &maxAM);
        max_buf_entry_size = (maxAM + 1) * (maxAM + 2) / 2;
        max_buf_entry_size = max_buf_entry_size * max_buf_entry_size;
        // max_buf_entry_size should be >= the product of any two items in {dimM, dimN, dimP, dimQ}
        update_F_buf_size  = 6 * max_buf_entry_size;
        int nthreads = omp_get_max_threads();
        update_F_buf = _mm_malloc(sizeof(double) * nthreads * update_F_buf_size, 64);
        assert(update_F_buf != NULL);
        
        if (ncpu_f == 1) use_atomic_add = 0;
    }
    
    init_block_buf(_nbf, _nshells, f_startind, num_dmat);
    
    #pragma omp parallel
    {
        // init    
        int nt = omp_get_thread_num();
        int nf = nt/ncpu_f;
        double *F_MN = &(F1[nf * sizeX1 * num_dmat]);
        double *F_PQ = &(F2[nf * sizeX2 * num_dmat]);
        double *F_NQ = F3;
        double *F_MP = &(F4[nf * sizeX4 * num_dmat]);
        double *F_MQ = &(F5[nf * sizeX5 * num_dmat]);
        double *F_NP = &(F6[nf * sizeX6 * num_dmat]);
        double mynsq = 0.0;
        double mynitl = 0.0;
        
        #pragma omp for
        for (int i = 0; i < nsp; i++) block_packed[i] = 0;
        
        // Pending quartets that need to be computed
        ThreadQuartetLists_s *thread_quartet_lists = (ThreadQuartetLists_s*) malloc(sizeof(ThreadQuartetLists_s));
        init_ThreadQuartetLists(thread_quartet_lists);
        
        // Simint multi_shellpair buffer for batch computation
        void *thread_multi_shellpair;
        CInt_SIMINT_createThreadMultishellpair(&thread_multi_shellpair);
        
        #pragma omp for schedule(dynamic) 
        for (int i = startMN; i < endMN; i++) 
        {
            int M = shellrid[i];
            int N = shellid[i];
            
            reset_ThreadQuartetLists(thread_quartet_lists, M, N);
            
            double value1 = shellvalue[i];            
            int dimM = f_startind[M + 1] - f_startind[M];
            int dimN = f_startind[N + 1] - f_startind[N];
            int iX1M = f_startind[M] - f_startind[startrow];
            int iX3M = rowpos[M]; 
            int iXN  = rowptr[i];
            int iMN  = iX1M * ldX1+ iXN;
            int flag1 = (value1 < 0.0) ? 1 : 0;   
            for (int j = startPQ; j < endPQ; j++) 
            {
                int P = shellrid[j];
                int Q = shellid[j];
                if ((M > P && (M + P) % 2 == 1) || 
                    (M < P && (M + P) % 2 == 0))
                    continue;                
                if ((M == P) &&
                    ((N > Q && (N + Q) % 2 == 1) ||
                    (N < Q && (N + Q) % 2 == 0)))
                    continue;
                double value2 = shellvalue[j];
                int dimP = f_startind[P + 1] - f_startind[P];
                int dimQ = f_startind[Q + 1] - f_startind[Q];
                int iX2P = f_startind[P] - f_startind[startcol];
                int iX3P = colpos[P];
                int iXQ  = colptr[j];               
                int iPQ  = iX2P * ldX2 + iXQ;                             
                int iNQ  = iXN  * ldX3 + iXQ;                
                int iMP  = iX1M * ldX4 + iX2P;
                int iMQ  = iX1M * ldX5 + iXQ;
                int iNP  = iXN  * ldX6 + iX2P;
                int iMP0 = iX3M * ldX3 + iX3P;
                int iMQ0 = iX3M * ldX3 + iXQ;
                int iNP0 = iXN  * ldX3 + iX3P;               
                int flag3 = (M == P && Q == N) ? 0 : 1;                    
                int flag2 = (value2 < 0.0) ? 1 : 0;
                if (fabs(value1 * value2) >= tolscr2) 
                {
                    mynsq  += 1.0;
                    mynitl += dimM * dimN * dimP * dimQ;
                    
                    // Pack D blocks
                    pack_D_block(M, N, dimM, dimN, D1[0] + iMN,  ldX1);
                    pack_D_block(P, Q, dimP, dimQ, D2[0] + iPQ,  ldX2);
                    pack_D_block(M, P, dimM, dimP, D3[0] + iMP0, ldX3);
                    pack_D_block(N, P, dimN, dimP, D3[0] + iNP0, ldX3);
                    pack_D_block(M, Q, dimM, dimQ, D3[0] + iMQ0, ldX3);
                    pack_D_block(N, Q, dimN, dimQ, D3[0] + iNQ,  ldX3);
                    
                    
                    // Save this shell pair to the target ket shellpair list
                    int am_pair_index = CInt_SIMINT_getShellpairAMIndex(simint, P, Q);
                    KetShellPairList_s *target_shellpair_list = &thread_quartet_lists->ket_shellpair_lists[am_pair_index];
                    int add_KetShellPair_ret = add_KetShellPair(
                        target_shellpair_list, P, Q,
                        dimM, dimN, dimP, dimQ, 
                        flag1, flag2, flag3,
                        iMN, iPQ, iMP, iNP, iMQ, iNQ,
                        iMP0, iMQ0, iNP0
                    );
                    assert(add_KetShellPair_ret == 1);
                    
                    // Target ket shellpair list is full, handles it
                    if (target_shellpair_list->num_shellpairs == _SIMINT_NSHELL_SIMD) 
                    {
                        int npairs = target_shellpair_list->num_shellpairs;
                        double *thread_batch_integrals;
                        int thread_batch_nints;
                        
                        CInt_computeShellQuartetBatch_SIMINT(
                            simint, nt,
                            thread_quartet_lists->M, 
                            thread_quartet_lists->N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            npairs, &thread_batch_integrals, &thread_batch_nints,
                            &thread_multi_shellpair
                        );
                        
                        if (thread_batch_nints > 0)
                        {
                            double st, et;
                            st = CInt_get_walltime_sec();
                            update_F_with_KetShellPairList(
                                nt, num_dmat, thread_batch_integrals, thread_batch_nints, npairs, 
                                target_shellpair_list,
                                D1, D2, D3,
                                F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
                                sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6,
                                ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, M, N
                            );
                            et = CInt_get_walltime_sec();
                            if (nt == 0) 
                                CInt_SIMINT_addupdateFtimer(simint, et - st);
                        }
                        
                        // Ket shellpair list is processed, reset it
                        reset_KetShellPairList(target_shellpair_list);
                    }
                }  // if (fabs(value1 * value2) >= tolscr2) 
            }  // for (int j = startPQ; j < endPQ; j++)
            
            // Process all the remaining shell pairs in the thread's list
            for (int am_pair_index = 0; am_pair_index < _SIMINT_AM_PAIRS; am_pair_index++)
            {
                KetShellPairList_s *target_shellpair_list = &thread_quartet_lists->ket_shellpair_lists[am_pair_index];
                
                if (target_shellpair_list->num_shellpairs > 0)  // Ket shellpair list is not empty, handles it
                {
                    int npairs = target_shellpair_list->num_shellpairs;
                    double *thread_batch_integrals;
                    int thread_batch_nints;
                    
                    CInt_computeShellQuartetBatch_SIMINT(
                        simint, nt,
                        thread_quartet_lists->M, 
                        thread_quartet_lists->N, 
                        target_shellpair_list->P_list,
                        target_shellpair_list->Q_list,
                        npairs, &thread_batch_integrals, &thread_batch_nints,
                        &thread_multi_shellpair
                    );
                    
                    if (thread_batch_nints > 0)
                    {
                        double st, et;
                        st = CInt_get_walltime_sec();
                        update_F_with_KetShellPairList(
                            nt, num_dmat, thread_batch_integrals, thread_batch_nints, npairs, 
                            target_shellpair_list,
                            D1, D2, D3,
                            F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
                            sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6,
                            ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, M, N
                        );
                        et = CInt_get_walltime_sec();
                        if (nt == 0) 
                            CInt_SIMINT_addupdateFtimer(simint, et - st);
                    }
                    
                    // Ket shellpair list is processed, reset it
                    reset_KetShellPairList(target_shellpair_list);
                }
            }
        }  // for (int i = startMN; i < endMN; i++)

        #pragma omp critical
        {
            *nitl += mynitl;
            *nsq += mynsq;
        }
        
        CInt_SIMINT_freeThreadMultishellpair(&thread_multi_shellpair);
        free_ThreadQuartetLists(thread_quartet_lists);
        
        free(thread_quartet_lists);
    } // #pragma omp parallel
}

void reset_F(int numF, int num_dmat, double *F1, double *F2, double *F3,
             double *F4, double *F5, double *F6,
             int sizeX1, int sizeX2, int sizeX3,
             int sizeX4, int sizeX5, int sizeX6)
{
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX1 * num_dmat; k++) {
            F1[k] = 0.0;    
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX2 * num_dmat; k++) {
            F2[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k <    1 * sizeX3 * num_dmat; k++) {
            F3[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX4 * num_dmat; k++) {
            F4[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX5 * num_dmat; k++) {
            F5[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX6 * num_dmat; k++) {
            F6[k] = 0.0;
        }
    }
}

static int block_low(int i, int n, int block_size)
{
    long long bs = block_size;
    long long _n = n;
    long long _i = i;
    bs *= _i;
    bs /= _n;
    int res = bs;
    return res;
}

void reduce_F(int numF, int num_dmat,
              double *F1, double *F2, double *F3,
              double *F4, double *F5, double *F6,
              int sizeX1, int sizeX2, int sizeX3,
              int sizeX4, int sizeX5, int sizeX6,
              int maxrowsize, int maxcolsize,
              int maxrowfuncs, int maxcolfuncs,
              int iX3M, int iX3P,
              int ldX3, int ldX4, int ldX5, int ldX6)
{
    int nthreads = omp_get_max_threads();
    #pragma omp parallel
    {
        int spos, epos;
        int tid = omp_get_thread_num();
        
        spos = block_low(tid,     nthreads, sizeX1 * num_dmat);
        epos = block_low(tid + 1, nthreads, sizeX1 * num_dmat);
        for (int p = 1; p < numF; p++)
        {
            #pragma simd
            for (int k = spos; k < epos; k++)
                F1[k] += F1[k + p * sizeX1 * num_dmat];
        }
        
        spos = block_low(tid,     nthreads, sizeX2 * num_dmat);
        epos = block_low(tid + 1, nthreads, sizeX2 * num_dmat);
        for (int p = 1; p < numF; p++)
        {
            #pragma simd
            for (int k = spos; k < epos; k++)
                F2[k] += F2[k + p * sizeX2 * num_dmat];
        }
        
        spos = block_low(tid,     nthreads, sizeX4 * num_dmat);
        epos = block_low(tid + 1, nthreads, sizeX4 * num_dmat);
        for (int p = 1; p < numF; p++)
        {
            #pragma simd
            for (int k = spos; k < epos; k++)
                F4[k] += F4[k + p * sizeX4 * num_dmat];
        }
        
        spos = block_low(tid,     nthreads, sizeX5 * num_dmat);
        epos = block_low(tid + 1, nthreads, sizeX5 * num_dmat);
        for (int p = 1; p < numF; p++)
        {
            #pragma simd
            for (int k = spos; k < epos; k++)
                F5[k] += F5[k + p * sizeX5 * num_dmat];
        }
        
        spos = block_low(tid,     nthreads, sizeX6 * num_dmat);
        epos = block_low(tid + 1, nthreads, sizeX6 * num_dmat);
        for (int p = 1; p < numF; p++)
        {
            #pragma simd
            for (int k = spos; k < epos; k++)
                F6[k] += F6[k + p * sizeX6 * num_dmat];
        }
        
        #pragma omp barrier

        int iMP = iX3M * ldX3 + iX3P;
        int iMQ = iX3M * ldX3;
        int iNP = iX3P;
        for (int k = 0; k < num_dmat; k++) {
            #pragma omp for
            for (int iM = 0; iM < maxrowfuncs; iM++) {
                for (int iP = 0; iP < maxcolfuncs; iP++) {
                    F3[k * sizeX3 + iMP + iM * ldX3 + iP]
                        += F4[k * sizeX4 + iM * ldX4 + iP];
                }
                for (int iQ = 0; iQ < maxcolsize; iQ++) {
                    F3[k * sizeX3 + iMQ + iM * ldX3 + iQ] +=
                        F5[k * sizeX5 + iM * ldX5 + iQ];    
                }
            }
            #pragma omp for
            for (int iN = 0; iN < maxrowsize; iN++) {
                for (int iP = 0; iP < maxcolfuncs; iP++) {
                    F3[k * sizeX3 + iNP + iN * ldX3 + iP] +=
                        F6[k * sizeX6 + iN * ldX6 + iP];
                }
            }
        }
    }
}
