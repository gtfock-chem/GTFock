#pragma once

// Split the 4-flood loop into 3 different 4-flood loop, so no 
// atomic_add_f64() is in the inner-most loop
static inline void update_F_split3(
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
        
        // for J_MN[imn], K_MP[imp], K_NP[inp]
        for (int iN = 0; iN < dimN; iN++) 
        {
            for (int iM = 0; iM < dimM; iM++) 
            {
                int imn = iM * ldMN + iN;
                double j_MN = 0.0, I;
                for (int iP = 0; iP < dimP; iP++) 
                {
                    int imp = iM * ldMP + iP;
                    int inp = iN * ldNP + iP;
                    double k_MP = 0.0, k_NP = 0.0;
                    
                    #pragma simd
                    for (int iQ = 0; iQ < dimQ; iQ++)   
                    {
                        I = integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];
                        j_MN += D_PQ[iP * ldPQ + iQ] * I;
                        k_MP -= D_NQ[iN * ldNQ + iQ] * I;
                        k_NP -= D_MQ[iM * ldNQ + iQ] * I;
                    }
                    
                    k_MP *= (1 + flag3) * 1.0;
                    k_NP *= (flag1 + flag5) * 1.0;
                    atomic_add_f64(&K_MP[imp], k_MP);
                    atomic_add_f64(&K_NP[inp], k_NP);
                }
                j_MN *= 2.0 * (1 + flag1 + flag2 + flag4);
                atomic_add_f64(&J_MN[imn], j_MN);
            }
        }
        
        // for K_MQ[imq], K_NQ[inq]
        for (int iM = 0; iM < dimM; iM++) 
        {
            for (int iQ = 0; iQ < dimQ; iQ++) 
            {
                int imq = iM * ldMQ + iQ;
                double k_MQ = 0.0, I;
                for (int iN = 0; iN < dimN; iN++) 
                {
                    int inq = iN * ldNQ + iQ;
                    double k_NQ = 0.0;
                    
                    #pragma simd
                    for (int iP = 0; iP < dimP; iP++) 
                    {
                        I = integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];
                        k_MQ -= D_NP[iN * ldNQ + iP] * I;
                        k_NQ -= D_MP[iM * ldNQ + iP] * I;
                    }
                    
                    k_NQ *= (flag4 + flag7) * 1.0;
                    atomic_add_f64(&K_NQ[inq], k_NQ);
                }
                k_MQ *= (flag2 + flag6) * 1.0;
                atomic_add_f64(&K_MQ[imq], k_MQ);
            }
        }
        
        // for J_PQ[ipq]
        for (int iP = 0; iP < dimP; iP++) 
        {
            for (int iQ = 0; iQ < dimQ; iQ++) 
            {
                int ipq = iP * ldPQ + iQ;
                double j_PQ = 0.0, I;
                for (int iM = 0; iM < dimM; iM++) 
                {
                    #pragma simd
                    for (int iN = 0; iN < dimN; iN++) 
                    {
                        I = integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];
                        j_PQ += D_MN[iM * ldMN + iN] * I;
                    }
                }
                j_PQ *= 2.0 * (flag3 + flag5 + flag6 + flag7);
                atomic_add_f64(&J_PQ[ipq], j_PQ);
            }
        }
        
    } // for (int i = 0 ; i < num_dmat; i++)
}

// Optimized update_F(): exchange 3rd and 5th loop, split 5th loop, reduce
// redundant computation, use thread-local buffer to reduce calling 
// atomic_add_f64(). Original update_F() is moved to the end of this file.
static inline void 
update_F_opt_iQ(
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
    
    double *K_NQ_buf = update_F_buf + 2 * update_F_buf_size * tid;
    double *J_PQ_buf = K_NQ_buf + update_F_buf_size;
    __assume_aligned(K_NQ_buf, 64);
    __assume_aligned(J_PQ_buf, 64);
    
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
    
        for (int iN = 0; iN < dimN; iN++) 
        {
            memset(K_NQ_buf, 0, sizeof(double) * dimQ);
            for (int iP = 0; iP < dimP; iP++) 
            {
                memset(J_PQ_buf, 0, sizeof(double) * dimQ);
                int inp = iN * ldNP + iP;
                double k_NP = 0.0;
                double vMQ  = (flag2 + flag6) * D_NP[iN * ldNQ + iP];
                for (int iM = 0; iM < dimM; iM++) 
                {
                    int imn = iM * ldMN + iN;
                    int imp = iM * ldMP + iP;
                    
                    int Ibase  = dimQ * (iP + dimP * (iN + dimN * iM));
                    double vPQ = D_MN[iM * ldMN + iN] * 2.0 * (flag3 + flag5 + flag6 + flag7);
                    double vNQ = D_MP[iM * ldNQ + iP] * (flag4 + flag7);
                    
                    double j_MN = 0.0, k_MP = 0.0, k_NP0 = 0.0;
                    
                    #pragma simd
                    for (int iQ = 0; iQ < dimQ; iQ++) 
                    {
                        double I = integrals[iQ + Ibase];
                        
                        j_MN  += D_PQ[iP * ldPQ + iQ] * I;
                        k_MP  -= D_NQ[iN * ldNQ + iQ] * I;
                        k_NP0 -= D_MQ[iM * ldNQ + iQ] * I;
                        
                        J_PQ_buf[iQ] += vPQ * I;
                        K_NQ_buf[iQ] -= vNQ * I;
                    }
                    j_MN *= 2.0 * (1 + flag1 + flag2 + flag4);
                    k_MP *= (1 + flag3);
                    k_NP += k_NP0 * (flag1 + flag5);
                    
                    for (int iQ = 0; iQ < dimQ; iQ++)
                    {
                        int imq = iM * ldMQ + iQ;
                        double I = integrals[iQ + Ibase];
                        atomic_add_f64(&K_MQ[imq], -vMQ * I);
                    }
                    
                    atomic_add_f64(&J_MN[imn], j_MN);
                    atomic_add_f64(&K_MP[imp], k_MP);
                }
                atomic_add_f64(&K_NP[inp], k_NP);
                for (int iQ = 0; iQ < dimQ; iQ++)
                    atomic_add_f64(&J_PQ[iP * ldPQ + iQ], J_PQ_buf[iQ]);
            } // for (int iP = 0; iP < dimP; iP++) 
            for (int iQ = 0; iQ < dimQ; iQ++)
                atomic_add_f64(&K_NQ[iN * ldNQ + iQ], K_NQ_buf[iQ]);
        } // for (int iN = 0; iN < dimN; iN++)
    } // for (int i = 0 ; i < num_dmat; i++)
}

// The slowest version, for showing the dependency of variables
static inline void update_F_naive(
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
    
        for (int iN = 0; iN < dimN; iN++) 
        {
            for (int iQ = 0; iQ < dimQ; iQ++) 
            {
                int inq = iN * ldNQ + iQ;
                for (int iM = 0; iM < dimM; iM++) 
                {
                    int imn = iM * ldMN + iN;
                    int imq = iM * ldMQ + iQ;
                    for (int iP = 0; iP < dimP; iP++) 
                    {
                        int ipq = iP * ldPQ + iQ;
                        int imp = iM * ldMP + iP;
                        int inp = iN * ldNP + iP;
                        
                        double I = integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))];

                        double vMN = 2.0 * (1 + flag1 + flag2 + flag4) * D_PQ[iP * ldPQ + iQ] * I;
                        atomic_add_f64(&J_MN[imn], vMN);

                        double vPQ = 2.0 * (flag3 + flag5 + flag6 + flag7) * D_MN[iM * ldMN + iN] * I;
                        atomic_add_f64(&J_PQ[ipq], vPQ);

                        double vMP = (1 + flag3) * 1.0 * D_NQ[iN * ldNQ + iQ] * I;
                        atomic_add_f64(&K_MP[imp], -vMP);

                        double vNP = (flag1 + flag5) * 1.0 * D_MQ[iM * ldNQ + iQ] * I;
                        atomic_add_f64(&K_NP[inp], -vNP);

                        double vMQ = (flag2 + flag6) * 1.0 * D_NP[iN * ldNQ + iP] * I;
                        atomic_add_f64(&K_MQ[imq], -vMQ);

                        double vNQ = (flag4 + flag7) * 1.0 * D_MP[iM * ldNQ + iP] * I;
                        atomic_add_f64(&K_NQ[inq], -vNQ);
                    }
                } // for (int iM = 0; iM < dimM; iM++) 
            } // for (int iQ = 0; iQ < dimQ; iQ++) 
        } // for (int iN = 0; iN < dimN; iN++)
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