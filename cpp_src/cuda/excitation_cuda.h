#pragma once
#include "../common/utils_cuda.h"
#include <cmath>

namespace squant {


__device__ inline void unpack_canon_cuda(int ij, int *s){
  int i = std::sqrt((ij+ 1) * 2) + 0.5;
  int j = ij - i*(i-1)/2;
  s[0] = i;
  s[1] = j;
}

int get_Num_SinglesDoubles_cuda(int sorb, int noA, int noB);

__device__ void unpack_SinglesDoubles_cuda(int sorb, int noA, int noB, int idx,
                                           int *idx_lst);

__device__ void get_comb_SD_cuda(unsigned long *bra, int *merged, int r0, int n,
                                 int len, int noa, int nob);

__device__ void get_comb_SD_cuda(unsigned long *bra, double *lst, int *merged,
                                 int r0, int n, int len, int noa, int nob);

} // namespace squant