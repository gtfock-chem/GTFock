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
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P
)
{
    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = 2 * ((dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN);
    assert(required_buf_size <= update_F_buf_size); 
    
    double *read_buf  = thread_buf;
    double *write_buf = thread_buf + (update_F_buf_size / 2);
    
    // Setup buffer pointers
    double *D_MN_buf = read_buf;  read_buf += dimM * dimN;
    double *D_MP_buf = read_buf;  read_buf += dimM * dimP;
    double *D_NP_buf = read_buf;  read_buf += dimN * dimP;
    double *D_PQ_buf = read_buf;  read_buf += dimP * dimQ;
    double *D_NQ_buf = read_buf;  read_buf += dimN * dimQ;
    double *D_MQ_buf = read_buf;  read_buf += dimM * dimQ;
    
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    
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
        if (load_MN)
        {
            for (int iM = 0; iM < dimM; iM++)
                memcpy(D_MN_buf + iM * dimN, D_MN + iM * ldMN, sizeof(double) * dimN);
            
            memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        }
        
        if (load_P)
        {
            for (int iM = 0; iM < dimM; iM++)
                memcpy(D_MP_buf + iM * dimP, D_MP + iM * ldNQ, sizeof(double) * dimP);
            
            for (int iN = 0; iN < dimN; iN++)
                memcpy(D_NP_buf + iN * dimP, D_NP + iN * ldNQ, sizeof(double) * dimP);
            
            memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        }
        
        
        for (int iM = 0; iM < dimM; iM++)
            memcpy(D_MQ_buf + iM * dimQ, D_MQ + iM * ldNQ, sizeof(double) * dimQ);
        
        for (int iN = 0; iN < dimN; iN++)
            memcpy(D_NQ_buf + iN * dimQ, D_NQ + iN * ldNQ, sizeof(double) * dimQ);
        
        for (int iP = 0; iP < dimP; iP++)
            memcpy(D_PQ_buf + iP * dimQ, D_PQ + iP * ldPQ, sizeof(double) * dimQ);
        
        // Reset result buffer
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
        if (write_MN)
        {
            for (int iM = 0; iM < dimM; iM++)
            {
                int iM_base1 = iM * dimN;
                int iM_base2 = iM * ldMN;
                for (int iN = 0; iN < dimN; iN++)
                {
                    double adder = J_MN_buf[iM_base1 + iN];
                    atomic_add_f64(&J_MN[iM_base2 + iN], adder);
                }
            }
        }

        if (write_P)
        {
            for (int iM = 0; iM < dimM; iM++)
            {
                int iM_base1 = iM * dimP;
                int iM_base2 = iM * ldMP;
                for (int iP = 0; iP < dimP; iP++)
                {
                    double adder = K_MP_buf[iM_base1 + iP];
                    atomic_add_f64(&K_MP[iM_base2 + iP], adder);
                }
            }

            for (int iN = 0; iN < dimN; iN++)
            {
                int iN_base1 = iN * dimP;
                int iN_base2 = iN * ldNP;
                for (int iP = 0; iP < dimP; iP++)
                {
                    double adder = K_NP_buf[iN_base1 + iP];
                    atomic_add_f64(&K_NP[iN_base2 + iP], adder);
                }
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

static inline void update_F_opt_buffer_Q1(
    int tid, int num_dmat, double *integrals, int dimM, int dimN,
    int dimP, int _dimQ,
    int flag1, int flag2, int flag3,
    int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
    int iMP0, int iMQ0, int iNP0,
    double **D1, double **D2, double **D3,
    double *F_MN, double *F_PQ, double *F_NQ,
    double *F_MP, double *F_MQ, double *F_NP,
    int sizeX1, int sizeX2, int sizeX3,
    int sizeX4, int sizeX5, int sizeX6,
    int ldMN, int ldPQ, int ldNQ, int ldMP, int ldMQ, int ldNP,
    int load_MN, int load_P, int write_MN, int write_P
)
{
    const int dimQ = 1;

    int flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    int flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    int flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    int flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;
    
    double *thread_buf = update_F_buf + tid * update_F_buf_size;
    int required_buf_size = 2 * ((dimP + dimN + dimM) * dimQ + (dimN + dimM) * dimP + dimM * dimN);
    assert(required_buf_size <= update_F_buf_size); 
    
    double *read_buf  = thread_buf;
    double *write_buf = thread_buf + (update_F_buf_size / 2);
    
    // Setup buffer pointers
    double *D_MN_buf = read_buf;  read_buf += dimM * dimN;
    double *D_MP_buf = read_buf;  read_buf += dimM * dimP;
    double *D_NP_buf = read_buf;  read_buf += dimN * dimP;
    double *D_PQ_buf = read_buf;  read_buf += dimP * dimQ;
    double *D_NQ_buf = read_buf;  read_buf += dimN * dimQ;
    double *D_MQ_buf = read_buf;  read_buf += dimM * dimQ;
    
    double *J_MN_buf = write_buf;  write_buf += dimM * dimN;
    double *K_MP_buf = write_buf;  write_buf += dimM * dimP;
    double *K_NP_buf = write_buf;  write_buf += dimN * dimP;
    double *J_PQ_buf = write_buf;  write_buf += dimP * dimQ;
    double *K_NQ_buf = write_buf;  write_buf += dimN * dimQ;
    double *K_MQ_buf = write_buf;  write_buf += dimM * dimQ;
    
    
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
        if (load_MN)
        {
            for (int iM = 0; iM < dimM; iM++)
                memcpy(D_MN_buf + iM * dimN, D_MN + iM * ldMN, sizeof(double) * dimN);
            
            memset(J_MN_buf, 0, sizeof(double) * dimM * dimN);
        }
        
        if (load_P)
        {
            for (int iM = 0; iM < dimM; iM++)
                memcpy(D_MP_buf + iM * dimP, D_MP + iM * ldNQ, sizeof(double) * dimP);
            
            for (int iN = 0; iN < dimN; iN++)
                memcpy(D_NP_buf + iN * dimP, D_NP + iN * ldNQ, sizeof(double) * dimP);
            
            memset(K_MP_buf, 0, sizeof(double) * dimP * (dimM + dimN));
        }
        
        
        for (int iM = 0; iM < dimM; iM++)
            memcpy(D_MQ_buf + iM * dimQ, D_MQ + iM * ldNQ, sizeof(double) * dimQ);
        
        for (int iN = 0; iN < dimN; iN++)
            memcpy(D_NQ_buf + iN * dimQ, D_NQ + iN * ldNQ, sizeof(double) * dimQ);
        
        for (int iP = 0; iP < dimP; iP++)
            memcpy(D_PQ_buf + iP * dimQ, D_PQ + iP * ldPQ, sizeof(double) * dimQ);
        
        // Reset result buffer
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
                    
                    int Ibase = iP + dimP * imn;
                    int ipq_base = iP * dimQ;
                    int imq_base = iM * dimQ;
                    int inq_base = iN * dimQ;
                    
                    double k_MN = 0.0, k_NP = 0.0;
                    
                    double I = integrals[Ibase];
                        
                    j_MN += D_PQ_buf[ipq_base] * I;
                    k_MN -= D_NQ_buf[inq_base] * I;
                    k_NP -= D_MQ_buf[imq_base] * I;
                    J_PQ_buf[ipq_base] += vPQ * I;
                    K_MQ_buf[imq_base] -= vMQ * I;
                    K_NQ_buf[inq_base] -= vNQ * I;
                    
                    K_MP_buf[imp] += k_MN * vMP_coef;
                    K_NP_buf[inp] += k_NP * vNP_coef;
                } // for (int iM = 0; iM < dimM; iM++) 
                J_MN_buf[imn] += j_MN * vMN_coef;
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
        
        // Update to the global array using atomic_add_f64()
        if (write_MN)
        {
            for (int iM = 0; iM < dimM; iM++)
            {
                int iM_base1 = iM * dimN;
                int iM_base2 = iM * ldMN;
                for (int iN = 0; iN < dimN; iN++)
                {
                    double adder = J_MN_buf[iM_base1 + iN];
                    atomic_add_f64(&J_MN[iM_base2 + iN], adder);
                }
            }
        }

        if (write_P)
        {
            for (int iM = 0; iM < dimM; iM++)
            {
                int iM_base1 = iM * dimP;
                int iM_base2 = iM * ldMP;
                for (int iP = 0; iP < dimP; iP++)
                {
                    double adder = K_MP_buf[iM_base1 + iP];
                    atomic_add_f64(&K_MP[iM_base2 + iP], adder);
                }
            }

            for (int iN = 0; iN < dimN; iN++)
            {
                int iN_base1 = iN * dimP;
                int iN_base2 = iN * ldNP;
                for (int iP = 0; iP < dimP; iP++)
                {
                    double adder = K_NP_buf[iN_base1 + iP];
                    atomic_add_f64(&K_NP[iN_base2 + iP], adder);
                }
            }
        }
        
        for (int iP = 0; iP < dimP; iP++)
        {
            int iP_base2 = iP * ldPQ;
            atomic_add_f64(&J_PQ[iP_base2], J_PQ_buf[iP]);
        }
        
        for (int iM = 0; iM < dimM; iM++)
        {
            int iM_base2 = iM * ldMQ;
            atomic_add_f64(&K_MQ[iM_base2], K_MQ_buf[iM]);
        }
        
        for (int iN = 0; iN < dimN; iN++)
        {
            int iN_base2 = iN * ldNQ;
            atomic_add_f64(&K_NQ[iN_base2], K_NQ_buf[iN]);
        }
        
    } // for (int i = 0 ; i < num_dmat; i++) 
}


// Won't speedup too much, ~2% for cc-pVDZ, < 1% for aug-cc-pVTZ and ANO-DZ
// Just for fulfilling my obsession :) 
static inline void update_F_1111(
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
    
    for (int i = 0 ; i < num_dmat; i++) 
    {
        double *D_MN = D1[i] + iMN;
        double *D_PQ = D2[i] + iPQ;
        double *D_NQ = D3[i] + iNQ;
        double *D_MP = D3[i] + iMP0;
        double *D_MQ = D3[i] + iMQ0;
        double *D_NP = D3[i] + iNP0;    

        double I = integrals[0];

        double vMN = 2.0 * (1 + flag1 + flag2 + flag4) * D_PQ[0] * I;
        double vPQ = 2.0 * (flag3 + flag5 + flag6 + flag7) * D_MN[0] * I;
        double vMP = (1 + flag3) * D_NQ[0] * I;
        double vNP = (flag1 + flag5) * D_MQ[0] * I;
        double vMQ = (flag2 + flag6) * D_NP[0] * I;
        double vNQ = (flag4 + flag7) * D_MP[0] * I;
        
        atomic_add_f64(&F_MN[i * sizeX1] + iMN, vMN);
        atomic_add_f64(&F_PQ[i * sizeX2] + iPQ, vPQ);
        atomic_add_f64(&F_MP[i * sizeX4] + iMP, -vMP);
        atomic_add_f64(&F_NP[i * sizeX6] + iNP, -vNP);
        atomic_add_f64(&F_MQ[i * sizeX5] + iMQ, -vMQ);
        atomic_add_f64(&F_NQ[i * sizeX3] + iNQ, -vNQ);
    } // for (int i = 0 ; i < num_dmat; i++)
}

// Original version can be found in the GitHub repository commit history
