
// Original implementation
// tid, load_MN, load_P, write_MN, write_P will not be used, just to 
// make the argument list the same as optimized version for convenience
static void update_F_orig(
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
    int load_MP, int load_P, int write_MN, int write_P
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
                        double I = integrals[iQ + dimQ*(iP + dimP * (iN + dimN * iM))]; //Simint  
                                 //integrals[iM + dimM*(iN + dimN * (iP + dimP * iQ))]; //OptERD
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

