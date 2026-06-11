#!/bin/bash
export OMP_NUM_THREADS=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((RANDOM + 20000))
export NPROC_PER_NODE=1

RUN_FILE=$1
# export TORCH_LOGS="dynamo,guards,graph_breaks,recompiles"
# export TORCH_LOGS="graph_breaks,recompiles
echo ${RUN_FILE}
echo "$0" "$@"
echo "==================run pytorch=================="
torchrun --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} $RUN_FILE
