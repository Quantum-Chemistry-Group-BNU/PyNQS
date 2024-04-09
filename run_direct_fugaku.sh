#!/bin/bash
export OMP_NUM_THREADS=12
export MASTER_ADDR="127.0.0.1"

port=$((RANDOM % (65535 - 49152 + 1) + 49152))
echo $port
export MASTER_PORT=$port

export NPROC_PER_NODE=1
export RUN_FILE="main.py"
echo "==================run pytorch=================="


# output_file="./debug-N2-fugaku-$port.txt"
# # 通过 exec 命令将后续执行的输出重定向到文件
# exec > "$output_file"  # 这将重定向标准输出
# exec 2>&1 


torchrun --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} $RUN_FILE
