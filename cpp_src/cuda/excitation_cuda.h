#pragma once
#include <cmath>

#include "utils_cuda.h"

namespace squant {

__device__ inline void unpack_canon_cuda(const int ij, int *s) {
  int i = std::sqrt((ij + 1) * 2) + 0.5;
  int j = ij - i * (i - 1) / 2;
  s[0] = i;
  s[1] = j;
}

__device__ __host__ int get_Num_SinglesDoubles_cuda(const int sorb, const int noA, const int noB);

__device__ void unpack_SinglesDoubles_cuda(const int sorb, const int noA,
                                           const int noB, const int idx,
                                           int *idx_lst);

__device__ void get_comb_SD_cuda(unsigned long *bra, const int *merged,
                                 const int r0, const int sorb, const int noA,
                                 const int noB);

__device__ void get_comb_SD_cuda(unsigned long *bra, double *lst,
                                 const int *merged, const int r0,
                                 const int sorb, const int noA, const int noB);

template <typename T>
__device__ T get_comb_SD_fused_cuda(unsigned long *comb, const int *merged,
                                         const T *h1e, const T *h2e,
                                         unsigned long *bra, const int r0,
                                         const int sorb, const int len,
                                         const int noA, const int noB);
}  // namespace squant