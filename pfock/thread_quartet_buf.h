#pragma once

// It seems that in the naming system of GTFock, *_t means pointer type,
// so I shall follow this way and use *_s to mark a struct type


// A KetShellPairList_s can holds _SIMINT_NSHELL_SIMD ket side shellpairs
// The ket side shellpairs should have same AM pairs
typedef struct 
{
    // Number of shell pairs in the list
    int num_shellpairs;
    
    // (P_list[i], Q_list[i]) are the shellpair ids for ket side
    // AM(P_list[]) are the same, AM(Q_list[]) are the same
    // fock_quartet_info are for calling update_F
    int *P_list, *Q_list, *fock_quartet_info;
    
    int *ptr;
} KetShellPairList_s;

typedef KetShellPairList_s* KetShellPairList_t;


// Different AM shellpair lists
typedef struct 
{
    // (M, N) are the shellpair id for bra side
    int M, N;
    
    // Shellpair lists for different AM pairs
    KetShellPairList_s *ket_shellpair_lists;  
    
    int *ptr;
} ThreadQuartetLists_s;

typedef ThreadQuartetLists_s* ThreadQuartetLists_t;


void init_KetShellPairList(KetShellPairList_s *ket_shellpair_list)
{
    assert(ket_shellpair_list != NULL);
    
    ket_shellpair_list->num_shellpairs = 0;
    
    ket_shellpair_list->ptr = (int*) malloc(sizeof(int) * _SIMINT_NSHELL_SIMD * (2 + 16));
    assert(ket_shellpair_list->ptr != NULL);
    
    ket_shellpair_list->P_list = ket_shellpair_list->ptr;
    ket_shellpair_list->Q_list = ket_shellpair_list->ptr + _SIMINT_NSHELL_SIMD;
    ket_shellpair_list->fock_quartet_info = ket_shellpair_list->ptr + _SIMINT_NSHELL_SIMD * 2;
}

void init_KetShellPairListwithBuffer(KetShellPairList_s *ket_shellpair_list, int *buffer)
{
    assert(ket_shellpair_list != NULL);
    
    ket_shellpair_list->num_shellpairs = 0;
    
    ket_shellpair_list->ptr = buffer;
    assert(ket_shellpair_list->ptr != NULL);
    
    ket_shellpair_list->P_list = ket_shellpair_list->ptr;
    ket_shellpair_list->Q_list = ket_shellpair_list->ptr + _SIMINT_NSHELL_SIMD;
    ket_shellpair_list->fock_quartet_info = ket_shellpair_list->ptr + _SIMINT_NSHELL_SIMD * 2;
}

void free_KetShellPairList(KetShellPairList_s *ket_shellpair_list)
{
    assert(ket_shellpair_list != NULL);
    
    ket_shellpair_list->num_shellpairs = 0;
    if (ket_shellpair_list->ptr != NULL) free(ket_shellpair_list->ptr);
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
    
    int *fock_info_list = ket_shellpair_list->fock_quartet_info + idx * 16;
    
    fock_info_list[0]  = _dimM;
    fock_info_list[1]  = _dimN;
    fock_info_list[2]  = _dimP;
    fock_info_list[3]  = _dimQ;
    fock_info_list[4]  = _flag1;
    fock_info_list[5]  = _flag2;
    fock_info_list[6]  = _flag3;
    fock_info_list[7]  = _iMN;
    fock_info_list[8]  = _iPQ;
    fock_info_list[9]  = _iMP;
    fock_info_list[10] = _iNP;
    fock_info_list[11] = _iMQ;
    fock_info_list[12] = _iNQ;
    fock_info_list[13] = _iMP0;
    fock_info_list[14] = _iMQ0;
    fock_info_list[15] = _iNP0;
    
    ket_shellpair_list->num_shellpairs++;
    return 1;
}

void init_ThreadQuartetLists(ThreadQuartetLists_s *thread_quartet_lists)
{
    assert(thread_quartet_lists != NULL);
    
    thread_quartet_lists->ket_shellpair_lists = (KetShellPairList_s *) malloc(sizeof(KetShellPairList_s) * _SIMINT_AM_PAIRS);  
    assert(thread_quartet_lists->ket_shellpair_lists != NULL);
    
    int spl_work_size = _SIMINT_NSHELL_SIMD * (2 + 16);
    int tql_work_size = spl_work_size * _SIMINT_AM_PAIRS;
    thread_quartet_lists->ptr = (int*) malloc(sizeof(int) * tql_work_size);
    assert(thread_quartet_lists->ptr != NULL);
    
    for (int i = 0; i < _SIMINT_AM_PAIRS; i++)
    {
        init_KetShellPairListwithBuffer(
            &thread_quartet_lists->ket_shellpair_lists[i],
            thread_quartet_lists->ptr + i * spl_work_size
        );
    }
}

void free_ThreadQuartetLists(ThreadQuartetLists_s *thread_quartet_lists)
{
    assert(thread_quartet_lists != NULL);
    
    assert(thread_quartet_lists->ket_shellpair_lists != NULL);
    for (int i = 0; i < _SIMINT_AM_PAIRS; i++)
        reset_KetShellPairList(&thread_quartet_lists->ket_shellpair_lists[i]);
    
    if (thread_quartet_lists->ket_shellpair_lists != NULL) 
        free(thread_quartet_lists->ket_shellpair_lists);
    
    if (thread_quartet_lists->ptr != NULL)
        free(thread_quartet_lists->ptr);
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

