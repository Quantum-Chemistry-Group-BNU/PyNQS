#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 仅由全局通信域中的第一个进程执行
    if (world_rank == 0) {
        // 删除旧的hostfile或创建一个新的空文件
        FILE* fp = fopen("tr_hostfile", "w");
        if (fp == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(fp); // 关闭文件即完成了清空或创建新文件的任务
    }

    // 确保所有进程等待直到hostfile已经被清空或创建
    MPI_Barrier(MPI_COMM_WORLD);

    char hostname[256];
    gethostname(hostname, 256);

    // 使用MPI_COMM_TYPE_SHARED创建一个每个节点上仅包含共享内存进程的通信器
    MPI_Comm shared_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);

    // 获取新通信器中的rank，每个节点上的共享内存进程将有自己的排名
    int shared_rank;
    MPI_Comm_rank(shared_comm, &shared_rank);

    if (shared_rank == 0) {  // 每个节点上的第一个进程
        // 仅第一个进程在每个节点上写入主机名
        FILE* fp = fopen("tr_hostfile", "a"); // 使用追加模式
        if (fp == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "%s\n", hostname);
        fclose(fp);
    }

    MPI_Comm_free(&shared_comm);
    MPI_Finalize();
    return 0;
}
