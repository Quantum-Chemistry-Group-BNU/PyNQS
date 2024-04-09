#!/bin/bash
#PJM -L  "node=2"
#PJM -L  "elapse=00:02:00"
#PJM --mpi "max-proc-per-node=2"
#PJM -g hp230257
#PJM --name xht

source ~/data/duyiming/venv/bin/activate
export OMP_NUM_THREADS=12

export NPROC_PER_NODE=1
export RUN_FILE="main.py"
HOST=`hostname`
echo ${HOST}
echo "env var"

# 编译MPI程序
mpifcc -o gather_hosts get_nodefile_mpi.c
# 运行MPI程序来收集节点信息并生成hostfile
mpirun ./gather_hosts
# master_addr=$(head -n 1 tr_hostfile)

HOSTFILE="tr_hostfile"
MASTER_ADDR=`head -n 1 ${HOSTFILE}`
HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=$HOST_RANK-1

echo "master addr:"
echo $MASTER_ADDR
# echo "NODE RANK:"
# echo $NODE_RANK
# echo "tr hostfile::"
# cat tr_hostfile
# printenv
# nodelist=$PJM_O_NODEINF
# echo $PJM_O_NODEINF
# cat $PJM_O_NODEINF
# echo "PJM rank"
# echo $PJM_RANK
# output_file="./debug-N2-fugaku-$port.txt"
# # 通过 exec 命令将后续执行的输出重定向到文件
# exec > "$output_file"  # 这将重定向标准输出
# exec 2>&1 


# torchrun --nnodes 2 --nproc_per_node ${NPROC_PER_NODE} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port 12334 $RUN_FILE
# torchrun --nnodes 2 --nproc_per_node ${NPROC_PER_NODE} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port 12334 $RUN_FILE
mpirun torchrun --nnodes 2 --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port 12334 $RUN_FILE
# mpirun torchrun --nnodes 2 --nproc_per_node ${NPROC_PER_NODE} $RUN_FILE
