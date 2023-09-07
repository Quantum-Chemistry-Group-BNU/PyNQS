#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
torchrun --nproc-per-node 4 main.py