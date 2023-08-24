#pragma once
#include "utils_cuda.h"
#include <cstddef>
#include <cstdint>

namespace squant {

__device__ inline int popcnt_cuda(const unsigned long x) { return __popcll(x); }
__device__ inline int get_parity_cuda(const unsigned long x) {
  return __popcll(x) & 1;
}
__device__ inline unsigned long get_ones_cuda(const int n) {
  return (1ULL << n) - 1ULL;
}  // parenthesis must be added due to priority
__device__ inline double num_parity_cuda(unsigned long x, int i) {
  // return 2.0f * static_cast<double>(x >> ( i - 1) & 1) - 1.0f;
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

__device__ inline int __ctzl(unsigned long x) {
  int r = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF) r -= 32;
  if (x & 0x0000FFFF0000FFFF) r -= 16;
  if (x & 0x00FF00FF00FF00FF) r -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F) r -= 4;
  if (x & 0x3333333333333333) r -= 2;
  if (x & 0x5555555555555555) r -= 1;
  return r;
}

__device__ inline int __clzl(unsigned long x) {
  int r = 0;
  if (!(x & 0xFFFFFFFF00000000)) r += 32, x <<= 32;
  if (!(x & 0xFFFF000000000000)) r += 16, x <<= 16;
  if (!(x & 0xFF00000000000000)) r += 8, x <<= 8;
  if (!(x & 0xF000000000000000)) r += 4, x <<= 4;
  if (!(x & 0xC000000000000000)) r += 2, x <<= 2;
  if (!(x & 0x8000000000000000)) r += 1, x <<= 1;
  return r;
}

__device__ void diff_type_cuda(const unsigned long *bra,
                               const unsigned long *ket, int *p,
                               const int _len);

__device__ int parity_cuda(const unsigned long *bra, const int n);

__device__ void diff_orb_cuda(const unsigned long *bra,
                              const unsigned long *ket, const int _len,
                              int *cre, int *ann);

__device__ void get_olst_cuda(const unsigned long *bra, int *olst,
                              const int _len);

__device__ void get_vlst_cuda(const unsigned long *bra, int *vlst, const int n,
                              const int _len);

__device__ void get_ovlst_cuda(const unsigned long *bra, int *merged,
                               const int sorb, const int nele,
                               const int bra_len);

// 0: unoccupied 1: occupied
__device__ void get_zvec_cuda(const unsigned long *bra, double *lst,
                              const int sorb, const int bra_len, const int idx);

__device__ int64_t permute_sgn_cuda(const int64_t *image2,
                                    const int64_t *onstate, 
                                    int64_t *index,
                                    const int size);

__device__ void sites_sym_index(const int64_t *onstate, const int nphysical,
                                const int64_t *data_index,
                                const int64_t *qrow_qcol,
                                const int64_t *qrow_qcol_index,
                                const int64_t *qrow_qcol_shape,
                                const int64_t *ista, const int64_t *ista_index,
                                const int64_t *image2, const int64_t nbatch,
                                int64_t *data_info, bool *sym_array);
}  // namespace squant