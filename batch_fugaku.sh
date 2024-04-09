#!/bin/bash
#PJM -L  "node=2"
#PJM -L  "elapse=00:02:00"
#PJM --mpi "max-proc-per-node=4"
#PJM -g hp230257
#PJM --name xht


RANK_SCRIPT="run_rank_fugaku.sh"
JOB_PATH="/home/u12219/scratch/xuhongtao/PyNQS-main"
NNODES=2
NODEFILE=nodelist_fugaku
echo `hostname`
touch ${NODEFILE}
cd ${JOB_PATH};sh ${RANK_SCRIPT} ${NNODES} ${NODEFILE}

# if [[ -f "${NODEFILE}" ]] ; then
#     rm ${NODEFILE}
# fi

#PJM -L  "rscgrp=small"
#PJM --mpi "shape=2x3x2"
#PJM -L  "freq=2200"