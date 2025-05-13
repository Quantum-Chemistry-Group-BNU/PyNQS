#include "../common/default.h"
#include "excitation_cuda.h"

#include "hamiltonian_cuda.h"
#include "onstate_cuda.h"
#include "stdio.h"

namespace squant {

__host__ __device__ int
get_Num_SinglesDoubles_cuda(const int sorb, const int noA, const int noB) {
  int k = sorb / 2;
  int nvA = k - noA, nvB = k - noB;
  int nSa = noA * nvA, nSb = noB * nvB;
  int nDaa = noA * (noA - 1) * nvA * (nvA - 1) / 4;
  int nDbb = noB * (noB - 1) * nvB * (nvB - 1) / 4;
  int nDab = noA * noB * nvA * nvB;
  return nSa + nSb + nDaa + nDbb + nDab;
}

__device__ void unpack_SinglesDoubles_cuda(const int sorb, const int noA,
                                            const int noB, const int idx,
                                            int *idx_lst) {
  int k = sorb / 2;
  int nvA = k - noA, nvB = k - noB;
  int nSa = noA * nvA, nSb = noB * nvB;
  int noAA = noA * (noA - 1) / 2;
  int noBB = noB * (noB - 1) / 2;
  int nvAA = nvA * (nvA - 1) / 2;
  int nvBB = nvB * (nvB - 1) / 2;
  int nDaa = noAA * nvAA;
  int nDbb = noBB * nvBB;
  int nDab = noA * noB * nvA * nvB;
  int dims[5] = {nSa, nSb, nDaa, nDbb, nDab};
  int d0 = dims[0];
  int d1 = dims[1] + d0;
  int d2 = dims[2] + d1;
  int d3 = dims[3] + d2;
  int i3 = idx >= d3;
  int i2 = idx >= d2;
  int i1 = idx >= d1;
  int i0 = idx >= d0;
  int icase = i0 + i1 + i2 + i3;
  int i, a, j, b;
  i = a = j = b = -1;
  int case_SD = 0;
  switch (icase) {
  case 0: {
    // aa
    int jdx = idx;
    i = 2 * (jdx % noA);
    a = 2 * (jdx / noA + noA); // alpha-even; beta-odd
    j = b = 0;
    break;
  }
  case 1: {
    // bb
    int jdx = idx - d0;
    i = 2 * (jdx % noB) + 1;
    a = 2 * (jdx / noB + noB) + 1;
    j = b = 0;
    break;
  }
  case 2: {
    // aaaa
    int jdx = idx - d1;
    int ijA = idx % noAA;
    int abA = jdx / noAA;
    int s1[2] = {0};
    int s2[2] = {0};
    unpack_canon_cuda(ijA, s1);
    unpack_canon_cuda(abA, s2);
    i = s1[0] * 2;
    j = s1[1] * 2;
    a = (s2[0] + noA) * 2;
    b = (s2[1] + noA) * 2;
    case_SD = 1;
    break;
  }
  case 3: {
    // bbbb
    int jdx = idx - d2;
    int ijB = idx % noBB;
    int abB = jdx / noBB;
    int s1[2] = {0};
    int s2[2] = {0};
    unpack_canon_cuda(ijB, s1);
    unpack_canon_cuda(abB, s2);
    i = s1[0] * 2 + 1; // i > j
    j = s1[1] * 2 + 1;
    a = (s2[0] + noB) * 2 + 1; // a > b
    b = (s2[1] + noB) * 2 + 1;
    case_SD = 1;
    break;
  }
  case 4: {
    // abab
    int jdx = idx - d3;
    int iaA = jdx % (noA * nvA);
    int jbB = jdx / (noA * nvA);
    i = (iaA % noA) * 2;
    a = (iaA / noA + noA) * 2;
    j = (jbB % noB) * 2 + 1;
    b = (jbB / noB + noB) * 2 + 1;
    case_SD = 1;
    break;
  }
  }
  idx_lst[0] = i;
  idx_lst[1] = a;
  idx_lst[2] = j;
  idx_lst[3] = b;
  idx_lst[4] = case_SD;
}

__device__ void get_comb_SD_cuda(unsigned long *comb, double *lst,
                                 const int *merged, const int r0,
                                 const int sorb, const int noA, const int noB) {
  int idx_lst[5] = {0};
  unpack_SinglesDoubles_cuda(sorb, noA, noB, r0, idx_lst);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    BIT_FLIP(comb[idx / 64], idx % 64);
    lst[idx] *= -1;
  }
}

template <typename T>
__device__ T get_comb_SD_fused_cuda(unsigned long *comb, const int *merged,
                                         const T *h1e, const T *h2e,
                                         unsigned long *bra, const int r0,
                                         const int sorb, const int len,
                                         const int noA, const int noB) {
  int idx_lst[5] = {0};
  int orbital_lst[4] = {0};
  int p[2], q[2];
  unsigned long ket[MAX_SORB_LEN] = {0};
  for(int i = 0; i < MAX_SORB_LEN; i++){
    ket[i] = bra[i];
  }

  unpack_SinglesDoubles_cuda(sorb, noA, noB, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    orbital_lst[i] = idx;
    BIT_FLIP(ket[idx / 64], idx % 64); // in-place
  }

  T Hij = 0.00;
  if (idx_lst[4] == 0) {
    // Single
    p[0] = orbital_lst[0];
    q[0] = orbital_lst[1];
    Hij += h1e_get_cuda(h1e, p[0], q[0], sorb); // hpq
    for (int i = 0; i < len; i++) {
      unsigned long repr = bra[i];
      while (repr != 0) {
        int j = 63 - __clzl(repr);
        int k = 64 * i + j;
        Hij += h2e_get_cuda(h2e, p[0], k, q[0], k); //<pk||qk>
        repr &= ~(1ULL << j);
      }
    }
    int sgn = parity_cuda(bra, p[0]) * parity_cuda(ket, q[0]);
    Hij *= static_cast<T>(sgn);
  } else {
    // double
    p[0] = max(orbital_lst[0], orbital_lst[2]);
    p[1] = min(orbital_lst[0], orbital_lst[2]);
    q[0] = max(orbital_lst[1], orbital_lst[3]);
    q[1] = min(orbital_lst[1], orbital_lst[3]);
    int sgn = parity_cuda(bra, p[0]) * parity_cuda(bra, p[1]) *
              parity_cuda(ket, q[0]) * parity_cuda(ket, q[1]);
    Hij = h2e_get_cuda(h2e, p[0], p[1], q[0], q[1]);
    Hij *= static_cast<T>(sgn);
    // printf("hij: %.4e\n", Hij);
  }
  for(int i = 0; i < MAX_SORB_LEN; i++){
    comb[i] = ket[i];
  }
  return Hij;
}

template __device__ double
get_comb_SD_fused_cuda<double>(unsigned long *comb, const int *merged,
                               const double *h1e, const double *h2e,
                               unsigned long *bra, const int r0, const int sorb,
                               const int len, const int noA, const int noB);

template __device__ float
get_comb_SD_fused_cuda<float>(unsigned long *comb, const int *merged,
                              const float *h1e, const float *h2e,
                              unsigned long *bra, const int r0, const int sorb,
                              const int len, const int noA, const int noB);

__device__ void get_comb_SD_cuda(unsigned long *comb, const int *merged,
                                 const int r0, const int sorb, const int noA,
                                 const int noB) {
  int idx_lst[5] = {0};
  unpack_SinglesDoubles_cuda(sorb, noA, noB, r0, idx_lst);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    BIT_FLIP(comb[idx / 64], idx % 64);
  }
}

} // namespace squant