#pragma once

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

static void atomic_update_block(double *dst, int ldd, double *src, int lds, int nrows, int ncols)
{
    for (int irow = 0; irow < nrows; irow++)
    {
        int dst_base = irow * ldd;
        int src_base = irow * lds;
        for (int icol = 0; icol < ncols; icol++)
            atomic_add_f64(&dst[dst_base + icol], src[src_base + icol]);
    }
}

static void direct_update_block(double *dst, int ldd, double *src, int lds, int nrows, int ncols)
{
    for (int irow = 0; irow < nrows; irow++)
    {
        int dst_base = irow * ldd;
        int src_base = irow * lds;
        // Don't vectorize this loop, its very short in most situation
        for (int icol = 0; icol < ncols; icol++)
            dst[dst_base + icol] += src[src_base + icol];
    }
}

static void update_global_blocks(
    int write_MN, int write_P, int dimM, int dimN, int dimP, int dimQ,
    int ldMN, int ldMP, int ldNP, int ldPQ, int ldMQ, int ldNQ,
    double *J_MN, double *J_MN_buf, double *K_MP, double *K_MP_buf,
    double *K_NP, double *K_NP_buf, double *J_PQ, double *J_PQ_buf,
    double *K_MQ, double *K_MQ_buf, double *K_NQ, double *K_NQ_buf
)
{
    if (use_atomic_add)
    {
        if (write_MN) atomic_update_block(J_MN, ldMN, J_MN_buf, dimN, dimM, dimN);

        if (write_P)
        {
            atomic_update_block(K_MP, ldMP, K_MP_buf, dimP, dimM, dimP);
            atomic_update_block(K_NP, ldNP, K_NP_buf, dimP, dimN, dimP);
        }
        
        atomic_update_block(J_PQ, ldPQ, J_PQ_buf, dimQ, dimP, dimQ);
        atomic_update_block(K_MQ, ldMQ, K_MQ_buf, dimQ, dimM, dimQ);
        atomic_update_block(K_NQ, ldNQ, K_NQ_buf, dimQ, dimN, dimQ);
    } else {
        if (write_MN) direct_update_block(J_MN, ldMN, J_MN_buf, dimN, dimM, dimN);

        if (write_P)
        {
            direct_update_block(K_MP, ldMP, K_MP_buf, dimP, dimM, dimP);
            direct_update_block(K_NP, ldNP, K_NP_buf, dimP, dimN, dimP);
        }
        
        direct_update_block(J_PQ, ldPQ, J_PQ_buf, dimQ, dimP, dimQ);
        direct_update_block(K_MQ, ldMQ, K_MQ_buf, dimQ, dimM, dimQ);
        atomic_update_block(K_NQ, ldNQ, K_NQ_buf, dimQ, dimN, dimQ);  // K_NQ always needs atomic update
    }
}

// Use thread-local buffer to reduce atomic add 
static inline void update_F_opt_buffer(
    int tid, int num_dmat, double *integrals, 
    int dimM, int dimN, int dimP, int dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    for (int i = 0; i < num_dmat; i++) 
    { 
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));

        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                int imn = iM * dimN + iN;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int inp = iN * dimP + iP;
                    int imp = iM * dimP + iP;
                    double vMQ = vMQ_coef * D_NP_buf[inp];
                    double vNQ = vNQ_coef * D_MP_buf[imp];
                    
                    int Ibase = dimQ * (iP + dimP * imn);
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    // dimQ is small, vectorizing short loops may hurt performance since
                    // it needs horizon reduction after the loop
                    for (int iQ = 0; iQ < dimQ; iQ++) 
                    {
                        double I = integrals[Ibase + iQ];
                        
                        j_MN += D_PQ_buf[ipq_base + iQ] * I;
                        k_MN -= D_NQ_buf[inq_base + iQ] * I;
                        k_NP -= D_MQ_buf[imq_base + iQ] * I;
                        J_PQ_buf[ipq_base + iQ] += vPQ * I;
                        K_MQ_buf[imq_base + iQ] -= vMQ * I;
                        K_NQ_buf[inq_base + iQ] -= vNQ * I;
                    }
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_opt_buffer_Q1(
    int tid, int num_dmat, double *integrals, 
    int dimM, int dimN, int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    const int dimQ = 1;

    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P) memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));
        
        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                const int imn = iM * dimN + iN;
                const int imn_dimP = imn * dimP;
                const int inp_base = iN * dimP;
                const int imp_base = iM * dimP;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0, k_MQ = 0.0, k_NQ = 0.0;
                // Don't vectorize this loop, too short
                for (int iP = 0; iP < dimP; iP++) 
                {
                    double vMQ = vMQ_coef * D_NP_buf[inp_base + iP];
                    double vNQ = vNQ_coef * D_MP_buf[imp_base + iP];
                    
                    double I = integrals[imn_dimP + iP];
                    
                    j_MN += I * D_PQ_buf[iP];
                    k_MQ -= vMQ * I;
                    k_NQ -= vNQ * I;
                    J_PQ_buf[iP * dimQ] += vPQ * I;
                    K_MP_buf[imp_base + iP] -= I * D_NQ_buf[iN] * vMP_coef;
                    K_NP_buf[inp_base + iP] -= I * D_MQ_buf[iM] * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[iM * dimN + iN] += j_MN * vMN_coef;
                K_MQ_buf[iM * dimQ] += k_MQ;
                K_NQ_buf[iN * dimQ] += k_NQ;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_opt_buffer_Q3(
    int tid, int num_dmat, double *integrals, 
    int dimM, int dimN, int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P, 
    int M, int N, int P, int Q
)
{
    const int dimQ = 3;
    
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));

        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                int imn = iM * dimN + iN;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int inp = iN * dimP + iP;
                    int imp = iM * dimP + iP;
                    double vMQ = vMQ_coef * D_NP_buf[inp];
                    double vNQ = vNQ_coef * D_MP_buf[imp];
                    
                    int Ibase = dimQ * (iP + dimP * imn);
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    #pragma unroll
                    for (int iQ = 0; iQ < 3; iQ++) 
                    {
                        double I = integrals[Ibase + iQ];
                        j_MN += D_PQ_buf[ipq_base + iQ] * I;
                        k_MN -= D_NQ_buf[inq_base + iQ] * I;
                        k_NP -= D_MQ_buf[imq_base + iQ] * I;
                        J_PQ_buf[ipq_base + iQ] += vPQ * I;
                        K_MQ_buf[imq_base + iQ] -= vMQ * I;
                        K_NQ_buf[inq_base + iQ] -= vNQ * I;
                    }
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_opt_buffer_Q6(
    int tid, int num_dmat, double *integrals, 
    int dimM, int dimN, int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    const int dimQ = 6;
    
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));

        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                int imn = iM * dimN + iN;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int inp = iN * dimP + iP;
                    int imp = iM * dimP + iP;
                    double vMQ = vMQ_coef * D_NP_buf[inp];
                    double vNQ = vNQ_coef * D_MP_buf[imp];
                    
                    int Ibase = dimQ * (iP + dimP * imn);
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    #pragma ivdep
                    for (int iQ = 0; iQ < 6; iQ++) 
                    {
                        double I = integrals[Ibase + iQ];
                        j_MN += D_PQ_buf[ipq_base + iQ] * I;
                        k_MN -= D_NQ_buf[inq_base + iQ] * I;
                        k_NP -= D_MQ_buf[imq_base + iQ] * I;
                        J_PQ_buf[ipq_base + iQ] += vPQ * I;
                        K_MQ_buf[imq_base + iQ] -= vMQ * I;
                        K_NQ_buf[inq_base + iQ] -= vNQ * I;
                    }
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_opt_buffer_Q10(
    int tid, int num_dmat, double *integrals,
    int dimM, int dimN, int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    const int dimQ = 10;
    
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));

        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                int imn = iM * dimN + iN;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int inp = iN * dimP + iP;
                    int imp = iM * dimP + iP;
                    double vMQ = vMQ_coef * D_NP_buf[inp];
                    double vNQ = vNQ_coef * D_MP_buf[imp];
                    
                    int Ibase = dimQ * (iP + dimP * imn);
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    #pragma ivdep
                    for (int iQ = 0; iQ < 10; iQ++) 
                    {
                        double I = integrals[Ibase + iQ];
                        j_MN += D_PQ_buf[ipq_base + iQ] * I;
                        k_MN -= D_NQ_buf[inq_base + iQ] * I;
                        k_NP -= D_MQ_buf[imq_base + iQ] * I;
                        J_PQ_buf[ipq_base + iQ] += vPQ * I;
                        K_MQ_buf[imq_base + iQ] -= vMQ * I;
                        K_NQ_buf[inq_base + iQ] -= vNQ * I;
                    }
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_opt_buffer_Q15(
    int tid, int num_dmat, double *integrals, 
    int dimM, int dimN, int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    const int dimQ = 15;
    
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = (dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN;
    assert(required_buf_size <= update_F_buf_size); 
    
    double *write_buf = thread_buf;
    
    // Setup buffer pointers
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
        
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];
    
        // Reset result buffer
        if (load_MN) memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        if (load_P)  memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        memset(J_PQ_buf, 0, sizeof(double) * dimQ * (dimM + dimN + dimP));

        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        double vMP_coef = (1 + flag3) * 1.0;
        double vNP_coef = (flag1 + flag5) * 1.0;

        // Start computation
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iN = 0; iN < dimN; iN++) 
            {
                int imn = iM * dimN + iN;
                double vPQ = vPQ_coef * D_MN_buf[imn];
                double j_MN = 0.0;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int inp = iN * dimP + iP;
                    int imp = iM * dimP + iP;
                    double vMQ = vMQ_coef * D_NP_buf[inp];
                    double vNQ = vNQ_coef * D_MP_buf[imp];
                    
                    int Ibase = dimQ * (iP + dimP * imn);
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    #pragma ivdep
                    for (int iQ = 0; iQ < 15; iQ++) 
                    {
                        double I = integrals[Ibase + iQ];
                        j_MN += D_PQ_buf[ipq_base + iQ] * I;
                        k_MN -= D_NQ_buf[inq_base + iQ] * I;
                        k_NP -= D_MQ_buf[imq_base + iQ] * I;
                        J_PQ_buf[ipq_base + iQ] += vPQ * I;
                        K_MQ_buf[imq_base + iQ] -= vMQ * I;
                        K_NQ_buf[inq_base + iQ] -= vNQ * I;
                    }
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        update_global_blocks(
            write_MN, write_P, dimM, dimN, dimP, dimQ, 
            ldMN, ldMP, ldNP, ldPQ, ldMQ, ldNQ,
            J_MN, J_MN_buf, K_MP, K_MP_buf, 
            K_NP, K_NP_buf, J_PQ, J_PQ_buf,
            K_MQ, K_MQ_buf, K_NQ, K_NQ_buf
        );
    } // for (int i = 0 ; i < num_dmat; i++) 
}

static inline void update_F_1111(
    int tid, int num_dmat, double *integrals,
    int dimM, int dimN, int dimP, int dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P,
    int M, int N, int P, int Q
)
{
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    for (int i = 0; i < num_dmat; i++) 
    {
        double *D_MN_buf = D_blocks + mat_block_ptr[M * nshells + N];
        double *D_PQ_buf = D_blocks + mat_block_ptr[P * nshells + Q];
        double *D_MP_buf = D_blocks + mat_block_ptr[M * nshells + P];
        double *D_NP_buf = D_blocks + mat_block_ptr[N * nshells + P];
        double *D_MQ_buf = D_blocks + mat_block_ptr[M * nshells + Q];
        double *D_NQ_buf = D_blocks + mat_block_ptr[N * nshells + Q];

        double I = integrals[0];

        double vMN = 2.0 * (1 + flag1 + flag2 + flag4) * D_PQ_buf[0] * I;
        double vPQ = 2.0 * (flag3 + flag5 + flag6 + flag7) * D_MN_buf[0] * I;
        double vMP = (1 + flag3) * D_NQ_buf[0] * I;
        double vNP = (flag1 + flag5) * D_MQ_buf[0] * I;
        double vMQ = (flag2 + flag6) * D_NP_buf[0] * I;
        double vNQ = (flag4 + flag7) * D_MP_buf[0] * I;
        
        atomic_add_f64(&F_MN[i * sizeX1] + iMN, vMN);
        atomic_add_f64(&F_PQ[i * sizeX2] + iPQ, vPQ);
        atomic_add_f64(&F_MP[i * sizeX4] + iMP, -vMP);
        atomic_add_f64(&F_NP[i * sizeX6] + iNP, -vNP);
        atomic_add_f64(&F_MQ[i * sizeX5] + iMQ, -vMQ);
        atomic_add_f64(&F_NQ[i * sizeX3] + iNQ, -vNQ);
    } // for (int i = 0 ; i < num_dmat; i++)
}

// See update_F_orig.h for the original implementation of update_F()

