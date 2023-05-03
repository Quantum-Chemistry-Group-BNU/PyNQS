#pragma once
#include "../common/utils_cuda.h"

namespace squant {

__device__ inline int popcnt_cuda(unsigned long x) { return __popcll(x); }
__device__ inline int get_parity_cuda(unsigned long x) { return __popcll(x) & 1; }
__device__ inline unsigned long get_ones_cuda(const int n) {
  return (1ULL << n) - 1ULL;
} // parenthesis must be added due to priority
__device__ inline double num_parity_cuda(unsigned long x, int i) {
  // return 2.0f * static_cast<double>(x >> ( i - 1) & 1) - 1.0f;
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

__device__ inline int __ctzl(unsigned long x) {
  int r = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF)
    r -= 32;
  if (x & 0x0000FFFF0000FFFF)
    r -= 16;
  if (x & 0x00FF00FF00FF00FF)
    r -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F)
    r -= 4;
  if (x & 0x3333333333333333)
    r -= 2;
  if (x & 0x5555555555555555)
    r -= 1;
  return r;
}

__device__ inline int __clzl(unsigned long x) {
  int r = 0;
  if (!(x & 0xFFFFFFFF00000000))
    r += 32, x <<= 32;
  if (!(x & 0xFFFF000000000000))
    r += 16, x <<= 16;
  if (!(x & 0xFF00000000000000))
    r += 8, x <<= 8;
  if (!(x & 0xF000000000000000))
    r += 4, x <<= 4;
  if (!(x & 0xC000000000000000))
    r += 2, x <<= 2;
  if (!(x & 0x8000000000000000))
    r += 1, x <<= 1;
  return r;
}

__device__ void diff_type_cuda(unsigned long *bra, unsigned long *ket, int *p, int _len);

__device__ int parity_cuda(unsigned long *bra, int n);

__device__ void diff_orb_cuda(unsigned long *bra, unsigned long *ket, int _len, int *cre,
                  int *ann);

__device__ void get_olst_cuda(unsigned long *bra, int *olst, int _len);

__device__ void get_vlst_cuda(unsigned long *bra, int *vlst, int n, int _len);

// vlst: abab
__device__ void get_vlst_cuda_ab(unsigned long *bra, int *vlst, int n, int _len);

// olst: abab
__device__ void get_olst_cuda_ab(unsigned long *bra, int *olst, int _len);


// -1: occupied 1: unoccupied
__device__ void get_zvec_cuda(unsigned long *bra, double *lst, const int sorb,
                         const int bra_len, const int idx);

} // namespace squant