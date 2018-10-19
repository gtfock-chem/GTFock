#include <stdlib.h>
#include <stdio.h>
//#include <ga.h>

#include "config.h"
#include "taskq.h"

#include "Buzz_Task_Queue.h"

int init_taskq(PFock_t pfock)
{
    Buzz_createBuzzTaskQueue(&pfock->task_queue, MPI_COMM_WORLD);
    return 0;
}


void clean_taskq(PFock_t pfock)
{
    Buzz_destroyBuzzTaskQueue(pfock->task_queue);
}


void reset_taskq(PFock_t pfock)
{
    Buzz_resetBuzzTaskQueue(pfock->task_queue);
}


int taskq_next(PFock_t pfock, int myrow, int mycol, int ntasks)
{
    int dst_rank  = myrow * pfock->npcol + mycol;

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    int next_task = Buzz_getNextTasks(pfock->task_queue, dst_rank, ntasks);
    gettimeofday(&tv2, NULL);
    pfock->timenexttask += (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;

    return next_task;
}
