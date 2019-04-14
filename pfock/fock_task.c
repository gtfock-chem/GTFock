#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
//#include <macdecls.h>
#include <sys/time.h>

#include "pfock.h"
#include "config.h"
#include "taskq.h"
#include "fock_task.h"
#include "cint_basisset.h"

// Using global variables is a bad habit, but it is convenient.
// Consider fix this problem later.

//#define DUP_F_PQ_BUF

// Array of thread quartet lists and the Simint multishellpair
#include "thread_quartet_buf.h"
ThreadQuartetLists_t *thread_quartet_listss;
void **thread_multi_shellpairs;

// update_F thread-local buffer
double *update_F_buf  = NULL;
int update_F_buf_size = 0;
int maxAM, max_dim, nthreads;

// Arrays for packed D and F storage
int    *mat_block_ptr;       // The offset of the 1st element of a block in the packed buffer
int    *F_PQ_blocks_to_F2;   // Mapping blocks in F_PQ_blocks to F2
int    *F_MNPQ_blocks_to_F3; // Mapping blocks in F_MNPQ_blocks to F3
int    *visited_Mpairs;      // Flags for marking if (M, i) is updated 
int    *visited_Npairs;      // Flags for marking if (N, i) is updated 
double *D_blocks;            // Packed density matrix (D) blocks
double *D_scrval;            // Maximum (in absolute) value of each D block
double *F_PQ_blocks;         // Packed F_PQ (J_PQ) blocks
double *F_MNPQ_blocks;       // Packed F_{MP, NP, MQ, NQ} (K_{MP, NP, MQ, NQ}) blocks
double *F_M_band_blocks;     // Thread-private buffer for F_MP and F_MQ blocks with the same M
double *F_N_band_blocks;     // Thread-private buffer for F_NP and F_NQ blocks with the same N

// Fixed pointers & values from PFock_t
BasisSet_t basis;
SIMINT_t   simint;
int    nbf, nshells, nsp, nbf2, F_PQ_block_size;
int    F_PQ_offset, myrank, maxcolfuncs, num_CPU_F, num_dup_F;
int    ncpu_f, num_dmat, sizeX1, sizeX2, sizeX3, ldX1, ldX2, ldX3;
int    *f_startind, *shell_bf_num; 
int    *shellptr, *shellid, *shellrid;
int    *rowpos, *colpos, *rowptr, *colptr;
int    *blkrowptr_sh, *blkcolptr_sh;
double tolscr2, *shellvalue, *D_mat, *F1, *nitl, *nsq;

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
    load_P, write_P, \
    M, N, P_list[ipair], Q_list[ipair], \
    thread_F_M_band_blocks, thread_M_bank_offset, \
    thread_F_N_band_blocks, thread_N_bank_offset, \
    thread_F_PQ_blocks

void update_F_with_KetShellPairList(
    int tid, int num_dmat, double *batch_integrals, int batch_nints, int npairs, 
    int M, int N, KetShellPairList_s *target_shellpair_list, 
    double *thread_F_M_band_blocks, double *thread_F_N_band_blocks
)
{
    int load_P, write_P;
    int same_P_s = 0, same_P_e = 0;
    int *P_list = target_shellpair_list->P_list;
    int *Q_list = target_shellpair_list->Q_list;
    int thread_M_bank_offset = mat_block_ptr[M * nshells];
    int thread_N_bank_offset = mat_block_ptr[N * nshells];
    #ifdef DUP_F_PQ_BUF
    double *thread_F_PQ_blocks = F_PQ_blocks + (tid / num_CPU_F) * F_PQ_block_size;
    #else
    double *thread_F_PQ_blocks = F_PQ_blocks;
    #endif
    
    int *fock_info_list = target_shellpair_list->fock_quartet_info;
    int is_1111 = fock_info_list[0] * fock_info_list[1] * fock_info_list[2] * fock_info_list[3];
    
    int curr_P = P_list[0];
    while (same_P_e < npairs)
    {
        for ( ; same_P_e < npairs; same_P_e++)
            if (curr_P != P_list[same_P_e]) break;
        
        for (int ipair = same_P_s; ipair < same_P_e; ipair++)
        {
            if (ipair == same_P_s) load_P = 1; 
            else load_P = 0;
            
            if (ipair == same_P_e - 1) write_P = 1;
            else write_P = 0;
            
            fock_info_list = target_shellpair_list->fock_quartet_info + ipair * 16;
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
        
        curr_P   = P_list[same_P_e];
        same_P_s = same_P_e;
    }
}

void init_block_buf(BasisSet_t _basis, PFock_t pfock)
{
    if (pfock->num_dmat != 1)
    {
        printf("  FATAL: currently JKD blocking only supports num_dmat==1 !! Please check scf.c\n");
        assert(num_dmat == 1);
    }

    if (update_F_buf_size > 0) return;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // Copy fixed pointers
    basis        = _basis;
    simint       = pfock->simint;
    ncpu_f       = pfock->ncpu_f;
    num_dmat     = pfock->num_dmat2;
    shellptr     = pfock->shellptr;
    shellvalue   = pfock->shellvalue;
    shellid      = pfock->shellid;
    shellrid     = pfock->shellrid;
    f_startind   = pfock->f_startind;
    rowpos       = pfock->rowpos;
    colpos       = pfock->colpos;
    rowptr       = pfock->rowptr;
    colptr       = pfock->colptr;
    tolscr2      = pfock->tolscr2;
    nbf          = pfock->nbf;
    nshells      = pfock->nshells;
    nsp          = nshells * nshells;
    nbf2         = nbf * nbf;
    maxcolfuncs  = pfock->maxcolfuncs;
    nthreads     = pfock->nthreads;
    D_mat        = pfock->D_mat;
    F1           = pfock->F1;
    nitl         = &pfock->uitl;
    nsq          = &pfock->usq;
    sizeX1       = pfock->sizeX1;
    sizeX2       = pfock->sizeX2;
    sizeX3       = pfock->sizeX3;
    ldX1         = pfock->maxrowsize;
    ldX2         = pfock->maxcolsize;
    ldX3         = pfock->maxcolsize;
    blkrowptr_sh = pfock->blkrowptr_sh;
    blkcolptr_sh = pfock->blkcolptr_sh;
    
    // Decide how many copies of F_PQ_blocks to use
    #ifdef DUP_F_PQ_BUF
    num_CPU_F = 1;
    num_dup_F = nthreads;
    #else
    num_CPU_F = nthreads;
    num_dup_F = 1;
    #endif
    if (myrank == 0)
    {
        if (num_CPU_F == 1) printf("  F_PQ_blocks won't use atomic add\n");
        else printf("  F_PQ_blocks will use atomic add\n");
    }
    
    // Allocate memory for blocked matrices
    F_PQ_block_size = nbf * maxcolfuncs;
    shell_bf_num  = (int*) malloc(sizeof(int) * nshells);
    mat_block_ptr = (int*) malloc(sizeof(int) * nsp);
    D_blocks      = (double*) malloc(sizeof(double) * nbf2);
    D_scrval      = (double*) malloc(sizeof(double) * nshells * nshells);
    F_PQ_blocks   = (double*) malloc(sizeof(double) * F_PQ_block_size * num_dup_F);
    F_MNPQ_blocks = (double*) malloc(sizeof(double) * nbf2);
    F_PQ_blocks_to_F2   = (int*) malloc(sizeof(int) * nsp);
    F_MNPQ_blocks_to_F3 = (int*) malloc(sizeof(int) * nsp);
    assert(mat_block_ptr != NULL);
    assert(shell_bf_num  != NULL);
    assert(D_blocks      != NULL);
    assert(D_scrval      != NULL);
    assert(F_PQ_blocks   != NULL);
    assert(F_MNPQ_blocks != NULL);
    assert(F_PQ_blocks_to_F2   != NULL);
    assert(F_MNPQ_blocks_to_F3 != NULL);
    double block_mem_MB = (double) nbf2 * 2 * sizeof(double);
    block_mem_MB += (double) nsp * 3 * sizeof(int);
    block_mem_MB /= 1048576.0;

    // Allocate memory for thread-local submatrices
    _maxMomentum(basis, &maxAM);
    max_dim = (maxAM + 1) * (maxAM + 2) / 2;
    F_M_band_blocks = (double*) malloc(sizeof(double) * nthreads * max_dim * nbf);
    F_N_band_blocks = (double*) malloc(sizeof(double) * nthreads * max_dim * nbf);
    visited_Mpairs  = (int*) malloc(sizeof(int) * nthreads * nshells);
    visited_Npairs  = (int*) malloc(sizeof(int) * nthreads * nshells);
    assert(F_M_band_blocks != NULL);
    assert(F_N_band_blocks != NULL);
    assert(visited_Mpairs  != NULL);
    assert(visited_Npairs  != NULL);
    double thread_buf_mem_MB = (double) nbf * 2 * (double) max_dim * sizeof(double);
    thread_buf_mem_MB += (double) nshells * 2 * sizeof(int);
    thread_buf_mem_MB *= (double) nthreads;
    thread_buf_mem_MB += (double) F_PQ_block_size * num_dup_F * sizeof(double);
    thread_buf_mem_MB /= 1048576.0;
    
    int max_buf_entry_size = max_dim * max_dim;
    update_F_buf_size = 6 * max_buf_entry_size;
    update_F_buf = _mm_malloc(sizeof(double) * nthreads * update_F_buf_size, 64);
    assert(update_F_buf != NULL);
    
    if (myrank == 0) 
    {
        printf("  Blocking matrix = %.2lf MB, ", block_mem_MB);
        printf("thread-local blocking buffer = %.2lf MB\n", thread_buf_mem_MB);
    }
    
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

    // Allocate and init each thread's shell quartet list and simint multi shellpair
    thread_quartet_listss   = (ThreadQuartetLists_t*) malloc(sizeof(ThreadQuartetLists_t) * nthreads);
    thread_multi_shellpairs = (void**) malloc(sizeof(void*) * nthreads);
    assert(thread_quartet_listss   != NULL);
    assert(thread_multi_shellpairs != NULL); 
    for (int i = 0; i < nthreads; i++)
    {
        thread_quartet_listss[i] = (ThreadQuartetLists_t) malloc(sizeof(ThreadQuartetLists_s));
        init_ThreadQuartetLists(thread_quartet_listss[i]);

        CInt_SIMINT_createThreadMultishellpair(&thread_multi_shellpairs[i]);
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

void pack_D_blocks()
{
    #pragma omp for 
    for (int M = 0; M < nshells; M++)
    {
        for (int N = 0; N < nshells; N++)
        {
            int dimM    = shell_bf_num[M];
            int dimN    = shell_bf_num[N];
            int MN_id   = M * nshells + N;
            int f_idx_M = f_startind[M];
            int f_idx_N = f_startind[N];
            double *D_src = D_mat    + f_idx_M * nbf + f_idx_N;
            double *D_dst = D_blocks + mat_block_ptr[MN_id];
            copy_matrix_block(D_dst, dimN, D_src, nbf, dimM, dimN);
            
            double maxval = 0.0;
            for (int i = 0; i < dimM * dimN; i++)
            {
                double absval = fabs(D_dst[i]);
                if (absval > maxval) maxval = absval;
            }
            D_scrval[MN_id] = maxval;
        }
    }
}

void mark_JK_with_KetShellPairList(
    int M, int N, int npairs, KetShellPairList_s *target_shellpair_list,
    double *D_mat, int *f_startind, int nbf, 
    int *thread_visited_Mpairs, int *thread_visited_Npairs
)
{
    int prev_P = -1;
    int *P_list = target_shellpair_list->P_list;
    int *Q_list = target_shellpair_list->Q_list;
    
    for (int ipair = 0; ipair < npairs; ipair++)
    {
        int *fock_info_list = target_shellpair_list->fock_quartet_info + ipair * 16;

        int P    = P_list[ipair];
        int Q    = Q_list[ipair];
        int iPQ  = fock_info_list[8];
        int iMP  = fock_info_list[9];
        int iNP  = fock_info_list[10];
        int iMQ  = fock_info_list[11];
        int iNQ  = fock_info_list[12];
        
        if (prev_P != P_list[ipair]) 
        {
            F_MNPQ_blocks_to_F3[M * nshells + P] = iMP;
            F_MNPQ_blocks_to_F3[N * nshells + P] = iNP;
            
            thread_visited_Mpairs[P] = 1;
            thread_visited_Npairs[P] = 1;
        }
        
          F_PQ_blocks_to_F2[P * nshells + Q] = iPQ;
        F_MNPQ_blocks_to_F3[M * nshells + Q] = iMQ;
        F_MNPQ_blocks_to_F3[N * nshells + Q] = iNQ;
        
        thread_visited_Mpairs[Q] = 1;
        thread_visited_Npairs[Q] = 1;
        
        prev_P = P_list[ipair];
    }
}

// for SCF, J = K
// Batched ERI version
void fock_task(
    int nblks_col, int sblk_row, int sblk_col, 
    int task, int startrow, int startcol, int repack_D
)
{
    int rowid   = task / nblks_col;
    int colid   = task % nblks_col;
    int startM  = blkrowptr_sh[sblk_row + rowid];
    int endM    = blkrowptr_sh[sblk_row + rowid + 1] - 1;
    int startP  = blkcolptr_sh[sblk_col + colid];
    int endP    = blkcolptr_sh[sblk_col + colid + 1] - 1;
    int startMN = shellptr[startM];
    int endMN   = shellptr[endM + 1];
    int startPQ = shellptr[startP];
    int endPQ   = shellptr[endP + 1];
    
    // For mapping the write position of F4, F5, F6 to F3
    int _iX3M = rowpos[startrow];
    int _iX3P = colpos[startcol];
    
    // startcol is the column start position of shells
    // This value should remains unchanged when consuming tasks from the same MPI proc
    F_PQ_offset = mat_block_ptr[startcol * nshells];
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double mynsq  = 0.0;
        double mynitl = 0.0;
        double st, et;
        
        double *thread_F_M_band_blocks = F_M_band_blocks + tid * nbf * max_dim;
        double *thread_F_N_band_blocks = F_N_band_blocks + tid * nbf * max_dim;
        int    *thread_visited_Mpairs  = visited_Mpairs  + tid * nshells;
        int    *thread_visited_Npairs  = visited_Npairs  + tid * nshells;
        
        if (repack_D) pack_D_blocks();
        
        // Pending quartets that need to be computed
        ThreadQuartetLists_t thread_quartet_lists = thread_quartet_listss[tid];
        
        // Simint multi_shellpair buffer for batch computation
        void *thread_multi_shellpair = thread_multi_shellpairs[tid];
        
        #pragma omp for schedule(dynamic) 
        for (int i = startMN; i < endMN; i++) 
        {
            int M = shellrid[i];
            int N = shellid[i];
            
            reset_ThreadQuartetLists(thread_quartet_lists, M, N);
            
            memset(thread_F_M_band_blocks, 0, sizeof(double) * nbf * max_dim);
            memset(thread_F_N_band_blocks, 0, sizeof(double) * nbf * max_dim);
            memset(thread_visited_Mpairs,  0, sizeof(int)    * nshells);
            memset(thread_visited_Npairs,  0, sizeof(int)    * nshells);
            
            double value1 = shellvalue[i];
            int dimM = shell_bf_num[M];
            int dimN = shell_bf_num[N];
            int iX1M = f_startind[M] - f_startind[startrow];
            int iX3M = rowpos[M]; 
            int iXN  = rowptr[i];
            int iMN  = iX1M * ldX1 + iXN;
            int flag1 = (value1 < 0.0) ? 1 : 0;
            
            double *thread_MN_buf = update_F_buf + tid * update_F_buf_size;
            memset(thread_MN_buf, 0, sizeof(double) * dimM * dimN);
            
            for (int j = startPQ; j < endPQ; j++)
            {
                int P = shellrid[j];
                int Q = shellid[j];
                if ((M > P && (M + P) % 2 == 1) || 
                    (M < P && (M + P) % 2 == 0)) continue;                
                if ((M == P) &&
                    ((N > Q && (N + Q) % 2 == 1) ||
                    (N < Q && (N + Q) % 2 == 0))) continue;
                
                double value2 = shellvalue[j];
                int dimP = shell_bf_num[P];
                int dimQ = shell_bf_num[Q];
                int iX2P = f_startind[P] - f_startind[startcol];
                int iX3P = colpos[P];
                int iXQ  = colptr[j];               
                int iPQ  = iX2P * ldX2 + iXQ;                             
                int iNQ  = iXN  * ldX3 + iXQ;                
                int iMP0 = iX3M * ldX3 + iX3P;
                int iMQ0 = iX3M * ldX3 + iXQ;
                int iNP0 = iXN  * ldX3 + iX3P;  

                int iMP_F3 = (iX1M * ldX3 + iX2P) + (_iX3M * ldX3 + _iX3P);
                int iNP_F3 = (iXN  * ldX3 + iX2P) + _iX3P;
                int iMQ_F3 = (iX1M * ldX3 + iXQ)  + (_iX3M * ldX3);
                
                int flag3 = (M == P && Q == N) ? 0 : 1;                    
                int flag2 = (value2 < 0.0) ? 1 : 0;
                
                double D_scrvals[6], Dval;
                D_scrvals[0] = fabs(D_scrval[M * nshells + N]);
                D_scrvals[1] = fabs(D_scrval[M * nshells + P]);
                D_scrvals[2] = fabs(D_scrval[M * nshells + Q]);
                D_scrvals[3] = fabs(D_scrval[N * nshells + P]);
                D_scrvals[4] = fabs(D_scrval[N * nshells + Q]);
                D_scrvals[5] = fabs(D_scrval[P * nshells + Q]);
                Dval = D_scrvals[0];
                for (int Dval_i = 1; Dval_i < 6; Dval_i++)
                    if (D_scrvals[Dval_i] > Dval) Dval = D_scrvals[Dval_i];
                
                if (fabs(value1 * value2 * Dval) >= tolscr2) 
                {
                    mynsq  += 1.0;
                    mynitl += dimM * dimN * dimP * dimQ;

                    // Save this shell pair to the target ket shellpair list
                    int am_pair_index = CInt_SIMINT_getShellpairAMIndex(simint, P, Q);
                    KetShellPairList_s *target_shellpair_list = &thread_quartet_lists->ket_shellpair_lists[am_pair_index];
                    int add_KetShellPair_ret = add_KetShellPair(
                        target_shellpair_list, P, Q,
                        dimM, dimN, dimP, dimQ, 
                        flag1, flag2, flag3,
                        iMN, iPQ, iMP_F3, iNP_F3, iMQ_F3, iNQ,
                        iMP0, iMQ0, iNP0
                    );
                    assert(add_KetShellPair_ret == 1);
                    
                    // Target ket shellpair list is full, handles it
                    if (target_shellpair_list->num_shellpairs == _SIMINT_NSHELL_SIMD) 
                    {
                        int npairs = target_shellpair_list->num_shellpairs;
                        double *thread_batch_integrals;
                        int thread_batch_nints;
                        
                        mark_JK_with_KetShellPairList(
                            M, N, npairs, target_shellpair_list,
                            D_mat, f_startind, nbf, 
                            thread_visited_Mpairs, thread_visited_Npairs
                        );
                        
                        CInt_computeShellQuartetBatch_SIMINT(
                            simint, tid,
                            thread_quartet_lists->M, 
                            thread_quartet_lists->N, 
                            target_shellpair_list->P_list,
                            target_shellpair_list->Q_list,
                            npairs, &thread_batch_integrals, &thread_batch_nints,
                            &thread_multi_shellpair
                        );
                        
                        if (thread_batch_nints > 0)
                        {
                            st = CInt_get_walltime_sec();
                            update_F_with_KetShellPairList(
                                tid, num_dmat, thread_batch_integrals, thread_batch_nints,
                                npairs, M, N, target_shellpair_list,
                                thread_F_M_band_blocks, thread_F_N_band_blocks
                            );
                            et = CInt_get_walltime_sec();
                            if (tid == 0) CInt_SIMINT_addupdateFtimer(simint, et - st);
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
                    
                    mark_JK_with_KetShellPairList(
                        M, N, npairs, target_shellpair_list,
                        D_mat, f_startind, nbf, 
                        thread_visited_Mpairs, thread_visited_Npairs
                    );
                    
                    CInt_computeShellQuartetBatch_SIMINT(
                        simint, tid,
                        thread_quartet_lists->M, 
                        thread_quartet_lists->N, 
                        target_shellpair_list->P_list,
                        target_shellpair_list->Q_list,
                        npairs, &thread_batch_integrals, &thread_batch_nints,
                        &thread_multi_shellpair
                    );
                    
                    if (thread_batch_nints > 0)
                    {
                        st = CInt_get_walltime_sec();
                        update_F_with_KetShellPairList(
                            tid, num_dmat, thread_batch_integrals, thread_batch_nints, 
                            npairs, M, N, target_shellpair_list,
                            thread_F_M_band_blocks, thread_F_N_band_blocks
                        );
                        et = CInt_get_walltime_sec();
                        if (tid == 0) CInt_SIMINT_addupdateFtimer(simint, et - st);
                    }
                    
                    // Ket shellpair list is processed, reset it
                    reset_KetShellPairList(target_shellpair_list);
                }
            }
            
            // Update F_MN block to F1 and F_{MP, NP, MQ, NQ} blocks to F_MNPQ_blocks
            st = CInt_get_walltime_sec();
            direct_add_block(F1 + iMN, ldX1, thread_MN_buf, dimN, dimM, dimN);
            int thread_M_bank_offset = mat_block_ptr[M * nshells];
            int thread_N_bank_offset = mat_block_ptr[N * nshells];
            for (int iPQ = 0; iPQ < nshells; iPQ++)
            {
                int dim_iPQ = shell_bf_num[iPQ];
                if (thread_visited_Mpairs[iPQ]) 
                {
                    int MPQ_block_ptr = mat_block_ptr[M * nshells + iPQ];
                    double *global_F_MNPQ_block_ptr   = F_MNPQ_blocks + MPQ_block_ptr;
                    double *thread_F_M_band_block_ptr = thread_F_M_band_blocks + MPQ_block_ptr - thread_M_bank_offset;
                    atomic_add_vector(global_F_MNPQ_block_ptr, thread_F_M_band_block_ptr, dimM * dim_iPQ);
                }
                if (thread_visited_Npairs[iPQ]) 
                {
                    int NPQ_block_ptr = mat_block_ptr[N * nshells + iPQ];
                    double *global_F_MNPQ_block_ptr   = F_MNPQ_blocks + NPQ_block_ptr;
                    double *thread_F_N_band_block_ptr = thread_F_N_band_blocks + NPQ_block_ptr - thread_N_bank_offset;
                    atomic_add_vector(global_F_MNPQ_block_ptr, thread_F_N_band_block_ptr, dimN * dim_iPQ);
                }
            }
            et = CInt_get_walltime_sec();
            if (tid == 0) CInt_SIMINT_addupdateFtimer(simint, et - st);
            
        }  // for (int i = startMN; i < endMN; i++)

        #pragma omp critical
        {
            *nitl += mynitl;
            *nsq  += mynsq;
        }
    } // #pragma omp parallel
}

void reset_F(int numF, int num_dmat, double *F1, double *F2, double *F3, int sizeX1, int sizeX2, int sizeX3)
{
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX1 * num_dmat; k++) F1[k] = 0.0;    
        
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX2 * num_dmat; k++) F2[k] = 0.0;
        
        #pragma omp for nowait
        for (int k = 0; k <    1 * sizeX3 * num_dmat; k++) F3[k] = 0.0;
        
        
        #pragma omp for nowait
        for (int i = 0; i < nsp; i++)
        {
            F_PQ_blocks_to_F2[i]   = -1;
            F_MNPQ_blocks_to_F3[i] = -1;
        }
        
        #pragma omp for nowait
        for (int i = 0; i < nbf2; i++)
        {
            F_MNPQ_blocks[i] = 0.0;
        }
        
        #pragma omp for nowait
        for (int i = 0; i < F_PQ_block_size * num_dup_F; i++)
            F_PQ_blocks[i]   = 0.0;
    }
}

static inline void add_Fxx_block_to_Fxx(
    int *Fxx_blocks_to_Fxx, int bid, 
    double *Fxx_blocks, double *Fxx, int ldFxx, int Fxx_block_offset
)
{
    if (Fxx_blocks_to_Fxx[bid] == -1) return;
    
    int dimM = shell_bf_num[bid / nshells];
    int dimN = shell_bf_num[bid % nshells];
    double *Fxx_ptr       = Fxx + Fxx_blocks_to_Fxx[bid];
    double *Fxx_block_ptr = Fxx_blocks + (mat_block_ptr[bid] - Fxx_block_offset);
    
    for (int irow = 0; irow < dimM; irow++)
    {
        double *Fxx_row       = Fxx_ptr + irow * ldFxx;
        double *Fxx_block_row = Fxx_block_ptr + irow * dimN;
        for (int icol = 0; icol < dimN; icol++)
            Fxx_row[icol] += Fxx_block_row[icol];
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

void reduce_F(double *F1, double *F2, double *F3, int maxrowsize, int maxcolsize, int ldX3, int ldX4, int ldX5, int ldX6)
{
    int nthreads = omp_get_max_threads();
    #pragma omp parallel 
    {
        int spos, epos;
        int tid = omp_get_thread_num();
        spos = block_low(tid,     nthreads, F_PQ_block_size);
        epos = block_low(tid + 1, nthreads, F_PQ_block_size);
        
        // Reduce all copies of F_PQ_blocks to the first copy
        for (int p = 1; p < num_dup_F; p++)
        {
            int offset = p * F_PQ_block_size;
            #pragma simd
            for (int k = spos; k < epos; k++)
                F_PQ_blocks[k] += F_PQ_blocks[offset + k];
        }
        
        #pragma omp barrier
        
        #pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < nsp; i++)
        {
            add_Fxx_block_to_Fxx(F_PQ_blocks_to_F2,   i, F_PQ_blocks,   F2, maxcolsize, F_PQ_offset);
            add_Fxx_block_to_Fxx(F_MNPQ_blocks_to_F3, i, F_MNPQ_blocks, F3, ldX3, 0);
        }
    }
}
