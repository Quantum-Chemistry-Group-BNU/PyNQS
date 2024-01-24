#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((RANDOM + 20000))
export NPROC_PER_NODE=1 
export RUN_FILE="main.py"
echo "==================run pytorch=================="
torchrun --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} $RUN_FILE
