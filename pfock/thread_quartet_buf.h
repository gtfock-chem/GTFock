#pragma once

// It seems that in the naming system of GTFock, *_t means pointer type,
// so I shall follow this way and use *_s to mark a struct type

typedef struct 
{
    int dimM, dimN, dimP, dimQ, flag1, flag2, flag3;
    int iMN, iPQ, iMP, iNP, iMQ, iNQ, iMP0, iMQ0, iNP0;
} FockQuartetInfo_s;

typedef FockQuartetInfo_s* FockQuartetInfo_t;

// A KetShellPairList_s can holds _SIMINT_NSHELL_SIMD ket side shellpairs
// The ket side shellpairs should have same AM pairs
typedef struct 
{
    // Number of shell pairs in the list
    int num_shellpairs;
    
    // (P_list[i], Q_list[i]) are the shellpair ids for ket side
    // AM(P_list[]) are the same, AM(Q_list[]) are the same
    int *P_list, *Q_list;
    
    // Other info about ket side shell pairs, for calling update_F
    FockQuartetInfo_s *fock_info_list;  
} KetShellPairList_s;

typedef KetShellPairList_s* KetShellPairList_t;


// Different AM shellpair lists
typedef struct 
{
    // (M, N) are the shellpair id for bra side
    int M, N;
    
    // Shellpair lists for different AM pairs
    KetShellPairList_s *ket_shellpair_lists;  
} ThreadQuartetLists_s;

typedef ThreadQuartetLists_s* ThreadQuartetLists_t;


void init_KetShellPairList(KetShellPairList_s *ket_shellpair_list)
{
    assert(ket_shellpair_list != NULL);
    
    ket_shellpair_list->num_shellpairs = 0;
    
    ket_shellpair_list->P_list = (int *) malloc(sizeof(int) * _SIMINT_NSHELL_SIMD);
    ket_shellpair_list->Q_list = (int *) malloc(sizeof(int) * _SIMINT_NSHELL_SIMD);
    ket_shellpair_list->fock_info_list = (FockQuartetInfo_s *) malloc(sizeof(FockQuartetInfo_s) * _SIMINT_NSHELL_SIMD);
    
    assert(ket_shellpair_list->P_list != NULL);
    assert(ket_shellpair_list->Q_list != NULL);
    assert(ket_shellpair_list->fock_info_list != NULL);
}

void free_KetShellPairList(KetShellPairList_s *ket_shellpair_list)
{
    assert(ket_shellpair_list != NULL);
    
    ket_shellpair_list->num_shellpairs = 0;
    
    if (ket_shellpair_list->P_list != NULL) free(ket_shellpair_list->P_list);
    if (ket_shellpair_list->Q_list != NULL) free(ket_shellpair_list->Q_list);
    if (ket_shellpair_list->fock_info_list != NULL) free(ket_shellpair_list->fock_info_list);
}


void reset_KetShellPairList(KetShellPairList_s *ket_shellpair_list)
{
    assert(ket_shellpair_list != NULL);
    ket_shellpair_list->num_shellpairs = 0;
}

int add_KetShellPair(
    KetShellPairList_s *ket_shellpair_list, int _P, int _Q, 
    int _dimM, int _dimN, int _dimP, int _dimQ,
    int _flag1, int _flag2, int _flag3, 
    int _iMN, int _iPQ, int _iMP, int _iNP, int _iMQ, int _iNQ, 
    int _iMP0, int _iMQ0, int _iNP0
)
{
    int idx = ket_shellpair_list->num_shellpairs;
    if (idx == _SIMINT_NSHELL_SIMD) return 0;  // List is full, failed
    
    ket_shellpair_list->P_list[idx] = _P;
    ket_shellpair_list->Q_list[idx] = _Q;
    
    ket_shellpair_list->fock_info_list[idx].dimM  = _dimM;
    ket_shellpair_list->fock_info_list[idx].dimN  = _dimN;
    ket_shellpair_list->fock_info_list[idx].dimP  = _dimP;
    ket_shellpair_list->fock_info_list[idx].dimQ  = _dimQ;
    ket_shellpair_list->fock_info_list[idx].flag1 = _flag1;
    ket_shellpair_list->fock_info_list[idx].flag2 = _flag2;
    ket_shellpair_list->fock_info_list[idx].flag3 = _flag3;
    ket_shellpair_list->fock_info_list[idx].iMN   = _iMN;
    ket_shellpair_list->fock_info_list[idx].iPQ   = _iPQ;
    ket_shellpair_list->fock_info_list[idx].iMP   = _iMP;
    ket_shellpair_list->fock_info_list[idx].iNP   = _iNP;
    ket_shellpair_list->fock_info_list[idx].iMQ   = _iMQ;
    ket_shellpair_list->fock_info_list[idx].iNQ   = _iNQ;
    ket_shellpair_list->fock_info_list[idx].iMP0  = _iMP0;
    ket_shellpair_list->fock_info_list[idx].iMQ0  = _iMQ0;
    ket_shellpair_list->fock_info_list[idx].iNP0  = _iNP0;
    
    ket_shellpair_list->num_shellpairs++;
    return 1;
}

void init_ThreadQuartetLists(ThreadQuartetLists_s *thread_quartet_lists)
{
    assert(thread_quartet_lists != NULL);
    
    thread_quartet_lists->ket_shellpair_lists = (KetShellPairList_s *) malloc(sizeof(KetShellPairList_s) * _SIMINT_AM_PAIRS);  
    assert(thread_quartet_lists->ket_shellpair_lists != NULL);
    
    for (int i = 0; i < _SIMINT_AM_PAIRS; i++)
        init_KetShellPairList(&thread_quartet_lists->ket_shellpair_lists[i]);
}

void free_ThreadQuartetLists(ThreadQuartetLists_s *thread_quartet_lists)
{
    assert(thread_quartet_lists != NULL);
    
    assert(thread_quartet_lists->ket_shellpair_lists != NULL);
    for (int i = 0; i < _SIMINT_AM_PAIRS; i++)
        free_KetShellPairList(&thread_quartet_lists->ket_shellpair_lists[i]);
    
    if (thread_quartet_lists->ket_shellpair_lists != NULL) 
        free(thread_quartet_lists->ket_shellpair_lists);
}

void reset_ThreadQuartetLists(ThreadQuartetLists_s *thread_quartet_lists, const int _M, const int _N)
{
    assert(thread_quartet_lists != NULL);
    
    thread_quartet_lists->M = _M;
    thread_quartet_lists->N = _N;
    
    assert(thread_quartet_lists->ket_shellpair_lists != NULL);
    for (int i = 0; i < _SIMINT_AM_PAIRS; i++)
        reset_KetShellPairList(&thread_quartet_lists->ket_shellpair_lists[i]);
}

