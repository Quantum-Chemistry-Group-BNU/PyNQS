#!/bin/bash
export OMP_NUM_THREADS=12
source ~/data/duyiming/venv/bin/activate


##Config nnodes node_rank master_addr
NNODES=$1
HOSTFILE=$2
HOST=`hostname`
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_ADDR=`head -n 1 ${HOSTFILE}`
HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=$HOST_RANK-1

echo "NODE RANK ${NODE_RANK}; NNODES ${NNODES}; MASTER ${MASTER_ADDR}"

MASTER_PORT=12345
NPROC_PER_NODE=4

RUN_FILE="main.py"

##Start trochrun
torchrun --nnodes ${NNODES} --nproc_per_node ${NPROC_PER_NODE} --node_rank ${NODE_RANK}  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} $RUN_FILE
