#pragma once

// Use thread-local buffer to reduce atomic add and unused cache line touch
// of the D_** arrays. Permute the loop order to provide continuous memory 
// access in the inner-most loop. 
static inline void update_F_opt_buffer(
    int tid, int num_dmat, double *integrals, int dimM, int dimN,
    int dimP, int dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    int iMP0, int iMQ0, int iNP0,
    double **D1, double **D2, double **D3,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP
)
{
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = 2 * ((dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN);
    assert(required_buf_size <= update_F_buf_size); 
    
    // Setup buffer pointers
    double *D_MN_buf = thread_buf;  thread_buf += dimM * dimN;
    double *D_PQ_buf = thread_buf;  thread_buf += dimP * dimQ;
    double *D_NQ_buf = thread_buf;  thread_buf += dimN * dimQ;
    double *D_MP_buf = thread_buf;  thread_buf += dimM * dimP;
    double *D_MQ_buf = thread_buf;  thread_buf += dimM * dimQ;
    double *D_NP_buf = thread_buf;  thread_buf += dimN * dimP;
    
    double *J_MN_buf = thread_buf;  thread_buf += dimM * dimN;
    double *J_PQ_buf = thread_buf;  thread_buf += dimP * dimQ;
    double *K_NQ_buf = thread_buf;  thread_buf += dimN * dimQ;
    double *K_MP_buf = thread_buf;  thread_buf += dimM * dimP;
    double *K_MQ_buf = thread_buf;  thread_buf += dimM * dimQ;
    double *K_NP_buf = thread_buf;  thread_buf += dimN * dimP;
    
    for (int i = 0 ; i < num_dmat; i++) 
    {
        double *D_MN = D1[i] + iMN;
        double *D_PQ = D2[i] + iPQ;
        double *D_NQ = D3[i] + iNQ;
        double *D_MP = D3[i] + iMP0;
        double *D_MQ = D3[i] + iMQ0;
        double *D_NP = D3[i] + iNP0;    
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
    
        // Load required D_MN, D_PQ, D_NQ, D_MP, D_MQ, D_NP to buffer
        for (int iM = 0; iM < dimM; iM++)
        {
            memcpy(D_MN_buf + iM * dimN, D_MN + iM * ldMN, sizeof(double) * dimN);
            memcpy(D_MP_buf + iM * dimP, D_MP + iM * ldNQ, sizeof(double) * dimP);
            memcpy(D_MQ_buf + iM * dimQ, D_MQ + iM * ldNQ, sizeof(double) * dimQ);
        }
        
        for (int iN = 0; iN < dimN; iN++)
        {
            memcpy(D_NQ_buf + iN * dimQ, D_NQ + iN * ldNQ, sizeof(double) * dimQ);
            memcpy(D_NP_buf + iN * dimP, D_NP + iN * ldNQ, sizeof(double) * dimP);
        }
        
        for (int iP = 0; iP < dimP; iP++)
            memcpy(D_PQ_buf + iP * dimQ, D_PQ + iP * ldPQ, sizeof(double) * dimQ);
        
        // Reset result buffer
        memset(J_MN_buf, 0, sizeof(double) * required_buf_size / 2);
        
        double vPQ_coef = 2.0 * (flag3 + flag5 + flag6 + flag7);
        double vMQ_coef = (flag2 + flag6) * 1.0;
        double vNQ_coef = (flag4 + flag7) * 1.0;
        
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
                    K_MP_buf[imp] += k_MN;
                    K_NP_buf[inp] += k_NP;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        
        double vMN_coef = 2.0 * (1 + flag1 + flag2 + flag4);
        for (int iM = 0; iM < dimM; iM++)
        {
            int iM_base1 = iM * dimN;
            int iM_base2 = iM * ldMN;
            for (int iN = 0; iN < dimN; iN++)
            {
                double adder = J_MN_buf[iM_base1 + iN] * vMN_coef;
                atomic_add_f64(&J_MN[iM_base2 + iN], adder);
            }
        }
        
        double vMP_coef = (1 + flag3) * 1.0;
        for (int iM = 0; iM < dimM; iM++)
        {
            int iM_base1 = iM * dimP;
            int iM_base2 = iM * ldMP;
            for (int iP = 0; iP < dimP; iP++)
            {
                double adder = K_MP_buf[iM_base1 + iP] * vMP_coef;
                atomic_add_f64(&K_MP[iM_base2 + iP], adder);
            }
        }
        
        double vNP_coef = (flag1 + flag5) * 1.0;
        for (int iN = 0; iN < dimN; iN++)
        {
            int iN_base1 = iN * dimP;
            int iN_base2 = iN * ldNP;
            for (int iP = 0; iP < dimP; iP++)
            {
                double adder = K_NP_buf[iN_base1 + iP] * vNP_coef;
                atomic_add_f64(&K_NP[iN_base2 + iP], adder);
            }
        }
        
        for (int iP = 0; iP < dimP; iP++)
        {
            int iP_base1 = iP * dimQ;
            int iP_base2 = iP * ldPQ;
            for (int iQ = 0; iQ < dimQ; iQ++)
                atomic_add_f64(&J_PQ[iP_base2 + iQ], J_PQ_buf[iP_base1 + iQ]);
        }
        
        for (int iM = 0; iM < dimM; iM++)
        {
            int iM_base1 = iM * dimQ;
            int iM_base2 = iM * ldMQ;
            for (int iQ = 0; iQ < dimQ; iQ++)
                atomic_add_f64(&K_MQ[iM_base2 + iQ], K_MQ_buf[iM_base1 + iQ]);
        }
        
        for (int iN = 0; iN < dimN; iN++)
        {
            int iN_base1 = iN * dimQ;
            int iN_base2 = iN * ldNQ;
            for (int iQ = 0; iQ < dimQ; iQ++)
                atomic_add_f64(&K_NQ[iN_base2 + iQ], K_NQ_buf[iN_base1 + iQ]);
        }
        
    } // for (int i = 0 ; i < num_dmat; i++) 
}

// Original version with comment
static inline void update_F_orig(
    int tid, int num_dmat, double *integrals, int dimM, int dimN,
    int dimP, int dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    int iMP0, int iMQ0, int iNP0,
    double **D1, double **D2, double **D3,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP
)
{
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;

    for (int i = 0 ; i < num_dmat; i++) {
        double *D_MN = D1[i] + iMN;
        double *D_PQ = D2[i] + iPQ;
        double *D_NQ = D3[i] + iNQ;
        double *D_MP = D3[i] + iMP0;
        double *D_MQ = D3[i] + iMQ0;
        double *D_NP = D3[i] + iNP0;    
        double *J_MN = &F_MN[i * sizeX1] + iMN;
        double *J_PQ = &F_PQ[i * sizeX2] + iPQ;
        double *K_NQ = &F_NQ[i * sizeX3] + iNQ;
        double *K_MP = &F_MP[i * sizeX4] + iMP;
        double *K_MQ = &F_MQ[i * sizeX5] + iMQ;
        double *K_NP = &F_NP[i * sizeX6] + iNP;
    
        for (int iN = 0; iN < dimN; iN++) {
            for (int iQ = 0; iQ < dimQ; iQ++) {
                int inq = iN * ldNQ + iQ;
                double k_NQ = 0;
                for (int iM = 0; iM < dimM; iM++) {
                    int imn = iM * ldMN + iN;
                    int imq = iM * ldMQ + iQ;
                    double j_MN = 0;
                    double k_MQ = 0;
                    for (int iP = 0; iP < dimP; iP++) {
                        int ipq = iP * ldPQ + iQ;
                        int imp = iM * ldMP + iP;
                        int inp = iN * ldNP + iP;
                        double I = 
                            integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];//Simint
                          //integrals[iM + dimM*(iN + dimN * (iP + dimP * iQ))];//OptERD
                        // F(m, n) += D(p, q) * 2 * I(m, n, p, q)
                        // F(n, m) += D(p, q) * 2 * I(n, m, p, q)
                        // F(m, n) += D(q, p) * 2 * I(m, n, q, p)
                        // F(n, m) += D(q, p) * 2 * I(n, m, q, p)
                        double vMN = 2.0 * (1 + flag1 + flag2 + flag4) *
                            D_PQ[iP * ldPQ + iQ] * I;
                        j_MN += vMN;
                        // F(p, q) += D(m, n) * 2 * I(p, q, m, n)
                        // F(p, q) += D(n, m) * 2 * I(p, q, n, m)
                        // F(q, p) += D(m, n) * 2 * I(q, p, m, n)
                        // F(q, p) += D(n, m) * 2 * I(q, p, n, m)
                        double vPQ = 2.0 * (flag3 + flag5 + flag6 + flag7) *
                            D_MN[iM * ldMN + iN] * I;
                        atomic_add_f64(&J_PQ[ipq], vPQ);
                        // F(m, p) -= D(n, q) * I(m, n, p, q)
                        // F(p, m) -= D(q, n) * I(p, q, m, n)
                        double vMP = (1 + flag3) *
                            1.0 * D_NQ[iN * ldNQ + iQ] * I;
                        atomic_add_f64(&K_MP[imp], -vMP);
                        // F(n, p) -= D(m, q) * I(n, m, p, q)
                        // F(p, n) -= D(q, m) * I(p, q, n, m)
                        double vNP = (flag1 + flag5) *
                            1.0 * D_MQ[iM * ldNQ + iQ] * I;
                        atomic_add_f64(&K_NP[inp], -vNP);
                        // F(m, q) -= D(n, p) * I(m, n, q, p)
                        // F(q, m) -= D(p, n) * I(q, p, m, n)
                        double vMQ = (flag2 + flag6) *
                            1.0 * D_NP[iN * ldNQ + iP] * I;
                        k_MQ -= vMQ;
                        // F(n, q) -= D(m, p) * I(n, m, q, p)
                        // F(q, n) -= D(p, m) * I(q, p, n, m)
                        double vNQ = (flag4 + flag7) *
                            1.0 * D_MP[iM * ldNQ + iP] * I;
                        k_NQ -= vNQ;
                    }
                    atomic_add_f64(&J_MN[imn], j_MN);
                    atomic_add_f64(&K_MQ[imq], k_MQ);
                }
                atomic_add_f64(&K_NQ[inq], k_NQ);
            }
        } // for (int iN = 0; iN < dimN; iN++)
    } // for (int i = 0 ; i < num_dmat; i++)
}