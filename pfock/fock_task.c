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


static inline void atomic_add_f64(volatile double* global_value, double addend)
{
    uint64_t expected_value, new_value;
    do {
        double old_value = *global_value;
        expected_value = _castf64_u64(old_value);
        new_value = _castf64_u64(old_value + addend);
    } while (!__sync_bool_compare_and_swap((volatile uint64_t*)global_value,
                                           expected_value, new_value));
}

double *update_F_buf  = NULL;
int update_F_buf_size = 0;

#include "update_F.h"

#include "thread_quartet_buf.h"

void update_F_with_KetShellPairList(
    int tid, int num_dmat, double *batch_integrals, int batch_nints, int npairs, 
    KetShellPairList_s *target_shellpair_list,
    double **D1, double **D2, double **D3,
    double *F_MN, double *F_PQ, double *F_NQ, double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3, int sizeX4, int sizeX5, int sizeX6,
    int ldX1, int ldX2, int ldX3, int ldX4, int ldX5, int ldX6
)
{
    for (int ipair = 0; ipair < npairs; ipair++)
    {
        int *fock_info_list = target_shellpair_list->fock_quartet_info + ipair * 16;
        update_F_split3(
            tid, num_dmat, &batch_integrals[ipair * batch_nints], 
            fock_info_list[0], 
            fock_info_list[1], 
            fock_info_list[2], 
            fock_info_list[3], 
            fock_info_list[4], 
            fock_info_list[5], 
            fock_info_list[6], 
            fock_info_list[7], 
            fock_info_list[8], 
            fock_info_list[9], 
            fock_info_list[10], 
            fock_info_list[11], 
            fock_info_list[12], 
            fock_info_list[13], 
            fock_info_list[14], 
            fock_info_list[15], 
            D1, D2, D3,
            F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
            sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6,
            ldX1, ldX2, ldX3, ldX4, ldX5, ldX6
        );
    }
}

// for SCF, J = K
void fock_task(BasisSet_t basis, SIMINT_t simint, int ncpu_f, int num_dmat,
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
               double *nitl, double *nsq)              
{
    int startMN = shellptr[startM];
    int endMN = shellptr[endM + 1];
    int startPQ = shellptr[startP];
    int endPQ = shellptr[endP + 1];
    
    if (update_F_buf_size == 0)
    {
        // startPQ & endPQ & f_startind remains unchanged for each call
        // So just allocate the buffer once
        for (int j = startPQ; j < endPQ; j++) 
        {
            int Q = shellid[j];
            int dimQ = f_startind[Q + 1] - f_startind[Q];
            if (dimQ > update_F_buf_size) update_F_buf_size = dimQ;
        }
        update_F_buf_size = (update_F_buf_size + 8) / 8 * 8;  // Align to 64 bytes
        int nthreads = omp_get_max_threads();
        update_F_buf = _mm_malloc(sizeof(double) * nthreads * 2 * update_F_buf_size, 64);
        assert(update_F_buf != NULL);
    }
    
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
                    mynsq += 1.0;
                    mynitl += dimM*dimN*dimP*dimQ;              
                    
                    int am_pair_index = CInt_SIMINT_getShellpairAMIndex(simint, P, Q);
                    
                    KetShellPairList_s *target_shellpair_list = &thread_quartet_lists->ket_shellpair_lists[am_pair_index];
                    
                    // Save this shell pair to the target ket shellpair list
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
                            basis, simint, nt,
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
                                ldX1, ldX2, ldX3, ldX4, ldX5, ldX6
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
                        basis, simint, nt,
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
                            ldX1, ldX2, ldX3, ldX4, ldX5, ldX6
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
    } /* #pragma omp parallel */
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
        for (int k = 0; k < sizeX3 * num_dmat; k++) {
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
    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 0; k < sizeX1 * num_dmat; k++) {
            for (int p = 1; p < numF; p++) {
                F1[k] += F1[k + p * sizeX1 * num_dmat];
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX2 * num_dmat; k++) {
            for (int p = 1; p < numF; p++) {
                F2[k] += F2[k + p * sizeX2 * num_dmat];
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX4 * num_dmat; k++) {
            for (int p = 1; p < numF; p++) {
                F4[k] += F4[k + p * sizeX4 * num_dmat];   
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX5 * num_dmat; k++) {
            for (int p = 1; p < numF; p++) {
                F5[k] += F5[k + p * sizeX5 * num_dmat];   
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX6 * num_dmat; k++) {
            for (int p = 1; p < numF; p++) {
                F6[k] += F6[k + p * sizeX6 * num_dmat];   
            }
        }

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
