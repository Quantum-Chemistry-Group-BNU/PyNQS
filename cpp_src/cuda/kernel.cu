#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifdef HIP
#include "hip/amd_detail/amd_device_functions.h"
#endif
#include "cuda_handle_error.h" // gcc 13 error, compile using gcc 11

#include "../common/default.h"
#include "kernel.h"

namespace squant {

__device__ inline int popcnt_cuda(const unsigned long x) { return __popcll(x); }
__device__ inline int get_parity_cuda(const unsigned long x) {
  return __popcll(x) & 1;
}
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

__device__ inline void unpack_canon_cuda(const int ij, int *s) {
  int i = std::sqrt((ij + 1) * 2) + 0.5;
  int j = ij - i * (i - 1) / 2;
  s[0] = i;
  s[1] = j;
}

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

__device__ void unpack_Singles_Doubles_cuda(const int sorb, const int noA,
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
    break;
  }
  }
  idx_lst[0] = i;
  idx_lst[1] = a;
  idx_lst[2] = j;
  idx_lst[3] = b;
}

__device__ void get_comb_SD_cuda(unsigned long *comb, double *lst,
                                 const int *merged, const int r0,
                                 const int sorb, const int noA, const int noB) {
  int idx_lst[4] = {0};
  unpack_Singles_Doubles_cuda(sorb, noA, noB, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    BIT_FLIP(comb[idx / 64], idx % 64);
    lst[idx] *= -1;
  }
}

__device__ void get_comb_SD_cuda(unsigned long *comb, const int *merged,
                                 const int r0, const int sorb, const int noA,
                                 const int noB) {
  int idx_lst[4] = {0};
  unpack_Singles_Doubles_cuda(sorb, noA, noB, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    BIT_FLIP(comb[idx / 64], idx % 64);
  }
}

__device__ void diff_type_cuda(const unsigned long *bra,
                               const unsigned long *ket, int *p,
                               const int _len) {
  unsigned long idiff, icre, iann;
  for (int i = _len - 1; i >= 0; i--) {
    idiff = bra[i] ^ ket[i];
    icre = idiff & bra[i];
    iann = idiff & ket[i];
    p[0] += popcnt_cuda(icre);
    p[1] += popcnt_cuda(iann);
  }
}

__device__ int parity_cuda(const unsigned long *bra, const int sorb) {
  int p = 0;
  for (int i = 0; i < sorb / 64; i++) {
    p ^= get_parity_cuda(bra[i]);
  }
  if (sorb % 64 != 0) {
    p ^= get_parity_cuda((bra[sorb / 64] & get_ones_cuda(sorb % 64)));
  }
  return -2 * p + 1;
}

__device__ void diff_orb_cuda(const unsigned long *bra,
                              const unsigned long *ket, const int _len,
                              int *cre, int *ann) {
  int idx_cre = 0;
  int idx_ann = 0;
  for (int i = _len - 1; i >= 0; i--) {
    unsigned long idiff = bra[i] ^ ket[i];
    unsigned long icre = idiff & bra[i];
    unsigned long iann = idiff & ket[i];
    while (icre != 0) {
      int j = 63 - __clzl(icre); // unsigned long
      cre[idx_cre] = i * 64 + j;
      icre &= ~(1ULL << j);
      idx_cre++;
    }
    while (iann != 0) {
      int j = 63 - __clzl(iann); // unsigned long
      ann[idx_ann] = i * 64 + j;
      iann &= ~(1ULL << j);
      idx_ann++;
    }
  }
}

__device__ void get_olst_cuda(const unsigned long *bra, int *olst,
                              const int _len) {
  unsigned long tmp;
  int idx = 0;
  // printf("tmp %d\n", bra[0]);
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __ctzl(tmp);
      olst[idx] = i * 64 + j;
      tmp &= ~(1ULL << j);
      idx++;
    }
  }
}

__device__ void get_olst_ab_cuda(const unsigned long *bra, int *olst,
                                 const int _len) {
  // abab
  unsigned long tmp;
  int idx = 0;
  int ida = 0;
  int idb = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __ctzl(tmp);
      int s = i * 64 + j;
      if (s & 1) {
        idb++;
        idx = 2 * idb - 1;
      } else {
        ida++;
        idx = 2 * (ida - 1);
      }
      olst[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_vlst_cuda(const unsigned long *bra, int *vlst,
                              const int sorb, const int _len) {
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cuda(sorb % 64));
    while (tmp != 0) {
      int j = __ctzl(tmp);
      vlst[ic] = i * 64 + j;
      ic++;
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_vlst_ab_cuda(const unsigned long *bra, int *vlst,
                                 const int sorb, const int _len) {
  // abab
  int ic = 0;
  int idb = 0;
  int ida = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cuda(sorb % 64));
    while (tmp != 0) {
      int j = __ctzl(tmp);
      int s = i * 64 + j;
      if (s & 1) {
        idb++;
        ic = 2 * idb - 1;
      } else {
        ida++;
        ic = 2 * (ida - 1);
      }
      vlst[ic] = s;
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_ovlst_cuda(const unsigned long *bra, int *merged,
                               const int sorb, const int nele,
                               const int bra_len) {
  get_olst_ab_cuda(bra, merged, bra_len);
  get_vlst_ab_cuda(bra, merged + nele, sorb, bra_len);
}

__device__ void get_zvec_cuda(const unsigned long *bra, double *lst,
                              const int sorb, const int bra_len,
                              const int idx) {
  constexpr int block = 64;
  const int idx_bra = idx / block;
  const int idx_bit = idx % block;
  lst[idx] = num_parity_cuda(bra[idx_bra], idx_bit + 1);
  ///
}

__device__ int64_t permute_sgn_cuda(const int64_t *image2,
                                    const int64_t *onstate, int64_t *index,
                                    const int size) {
  int64_t sgn = 0;
  for (int i = 0; i < size; i++) {
    if (image2[i] == index[i]) {
      continue;
    }
    // find the position of target image2[i] in index
    int k = 0;
    for (int j = i + 1; j < size; j++) {
      if (index[j] == image2[i]) {
        k = j;
        break;
      }
    }
    // shift data
    bool fk = onstate[index[k]];
    for (int j = k - 1; j >= i; j--) {
      index[j + 1] = index[j];
      if (fk && onstate[index[j]]) {
        sgn ^= 1;
      }
    }
    index[i] = image2[i];
  }
  return -2 * sgn + 1;
}

__device__ inline void binary_search_cuda(const int64_t *arr,
                                          const int64_t *target,
                                          const int64_t length, int64_t *result,
                                          const int stride = 4) {
  int64_t left = 0;
  int64_t right = length / stride - 1;

  int64_t value_1 = -1;
  int64_t value_2 = -1;

  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    int64_t mid_index = mid * stride;
    int64_t mid_x1 = arr[mid_index];
    int64_t mid_x2 = arr[mid_index + 1];

    if (mid_x1 == target[0] && mid_x2 == target[1]) {
      value_1 = arr[mid_index + 2];
      value_2 = arr[mid_index + 3];
      break;
    } else if (mid_x1 < target[0] ||
               (mid_x1 == target[0] && mid_x2 < target[1])) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  result[0] = value_1;
  result[1] = value_2;
}

__device__ void sites_sym_index(const int64_t *onstate, const int nphysical,
                                const int64_t *data_index,
                                const int64_t *qrow_qcol,
                                const int64_t *qrow_qcol_index,
                                const int64_t *qrow_qcol_shape,
                                const int64_t *ista, const int64_t *ista_index,
                                const int64_t *image2, const int64_t nbatch,
                                int64_t *data_info, bool *sym_array) {
  int64_t qsym_out[2] = {0, 0};
  int64_t qsym_in[2] = {0, 0};
  int64_t qsym_n[2] = {0, 0};
  bool qsym_break = false;

  // binary search
  int64_t begin = 0;
  int64_t end = 0;
  int64_t length = 0;
  int64_t result[2] = {0, 0};
  for (int i = nphysical - 1; i >= 0; i--) {
    const int64_t na = onstate[image2[2 * i]];
    const int64_t nb = onstate[image2[2 * i + 1]];

    int64_t idx = 0;
    if (na == 0 and nb == 0) { // 00
      idx = 0;
      qsym_n[0] = 0;
      qsym_n[1] = 0;
    } else if (na == 1 and nb == 1) { // 11
      idx = 1;
      qsym_n[0] = 2;
      qsym_n[1] = 0;
    } else if (na == 1 and nb == 0) { // a
      idx = 2;
      qsym_n[0] = 1;
      qsym_n[1] = 1;

    } else if (na == 0 and nb == 1) { // b
      idx = 3;
      qsym_n[0] = 1;
      qsym_n[1] = -1;
    }
    qsym_in[0] = qsym_out[0];
    qsym_in[1] = qsym_out[1];
    qsym_out[0] = qsym_in[0] + qsym_n[0];
    qsym_out[1] = qsym_in[1] + qsym_n[1];

    begin = qrow_qcol_index[i];
    end = qrow_qcol_index[i + 1];
    length = (end - begin) * 4;
    // XXX: dose not use template? compilation way error??
    binary_search_cuda(&qrow_qcol[begin * 4], qsym_out, length, result);
    int64_t dr = result[0];
    int64_t qi = result[1];

    // printf("(%ld, %ld, %ld)-1\n", begin, end, length);
    // for(int i = 0; i < length ; i++){
    //   printf("%ld ", qrow_qcol[begin * 4 + i]);
    // }
    // printf("\n");

    begin = qrow_qcol_index[i + 1];
    end = qrow_qcol_index[i + 2];
    length = (end - begin) * 4;
    binary_search_cuda(&qrow_qcol[begin * 4], qsym_in, length, result);
    int64_t dc = result[0];
    int64_t qj = result[1];

    // printf("(%ld, %ld, %ld)-2\n", begin, end, length);
    // for(int i = 0; i < length ; i++){
    //   printf("%ld ", qrow_qcol[begin * 4 + i]);
    // }
    // printf("\n");
    // printf("(%ld %ld), (%ld, %ld)\n", dr, dc, qi, qj);

    int64_t data_idx = data_index[i * 4 + idx];
    // [qi, qj], shape: (qrow, qcol)
    int64_t offset = qi * qrow_qcol_shape[i + 1] + qj;
    // ista[qi, qj]
    int ista_value = ista[ista_index[i * 4 + idx] + offset];
    if (qi == -1 or qj == -1 or ista_value == -1) {
      qsym_break = true;
      break;
    } else {
      data_idx += ista_value;
    }

    // data_info: (3, nphysical, nbatch)
    // slice: 0-dim:
    // data_info: (3, nphysical, batch]
    data_info[i * nbatch] = data_idx;
    data_info[i * nbatch + nbatch * nphysical * 1] = dr;
    data_info[i * nbatch + nbatch * nphysical * 2] = dc;
  }
  sym_array[0] = qsym_break;
}

__device__ double h1e_get_cuda(const double *h1e, const size_t i,
                               const size_t j, const size_t sorb) {
  return h1e[j * sorb + i];
}

__device__ double h2e_get_cuda(const double *h2e, const size_t i,
                               const size_t j, const size_t k, const size_t l) {
  if ((i == j) || (k == l))
    return 0.00;
  size_t ij = i > j ? i * (i - 1) / 2 + j : j * (j - 1) / 2 + i;
  size_t kl = k > l ? k * (k - 1) / 2 + l : l * (l - 1) / 2 + k;
  double sgn = 1;
  sgn = i > j ? sgn : -sgn;
  sgn = k > l ? sgn : -sgn;
  double val;
  if (ij >= kl) {
    size_t ijkl = ij * (ij + 1) / 2 + kl;
    val = sgn * h2e[ijkl];
  } else {
    size_t ijkl = kl * (kl + 1) / 2 + ij;
    val = sgn * h2e[ijkl]; // sgn * conjugate(h2e[ijkl])
  }
  return val;
}

__device__ double get_Hii_cuda(const unsigned long *bra,
                               const unsigned long *ket, const double *h1e,
                               const double *h2e, const size_t sorb,
                               const int nele, const int bra_len) {
  double Hii = 0.00;
  int olst[MAX_NELE] = {0};
  get_olst_cuda(bra, olst, bra_len);

  for (int i = 0; i < nele; i++) {
    int p = olst[i]; //<p|h|p>
    Hii += h1e_get_cuda(h1e, p, p, sorb);
    for (int j = 0; j < i; j++) {
      int q = olst[j];
      Hii += h2e_get_cuda(h2e, p, q, p, q); //<pq||pq> Storage not continuous
    }
  }
  return Hii;
}

__device__ double get_HijS_cuda(const unsigned long *bra,
                                const unsigned long *ket, const double *h1e,
                                const double *h2e, const size_t sorb,
                                const int bra_len) {
  double Hij = 0.00;
  int p[1], q[1];
  diff_orb_cuda(bra, ket, bra_len, p, q);
  Hij += h1e_get_cuda(h1e, p[0], q[0], sorb); // hpq
  for (int i = 0; i < bra_len; i++) {
    unsigned long repr = bra[i];
    while (repr != 0) {
      int j = 63 - __clzl(repr);
      int k = 64 * i + j;
      Hij += h2e_get_cuda(h2e, p[0], k, q[0], k); //<pk||qk>
      repr &= ~(1ULL << j);
    }
  }
  int sgn = parity_cuda(bra, p[0]) * parity_cuda(ket, q[0]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

__device__ double get_HijD_cuda(const unsigned long *bra,
                                const unsigned long *ket, const double *h1e,
                                const double *h2e, const size_t sorb,
                                const int bra_len) {
  int p[2], q[2];
  diff_orb_cuda(bra, ket, bra_len, p, q);
  int sgn = parity_cuda(bra, p[0]) * parity_cuda(bra, p[1]) *
            parity_cuda(ket, q[0]) * parity_cuda(ket, q[1]);
  double Hij = h2e_get_cuda(h2e, p[0], p[1], q[0], q[1]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

__device__ double get_Hij_cuda(const unsigned long *bra,
                               const unsigned long *ket, const double *h1e,
                               const double *h2e, const size_t sorb,
                               const int nele, const int bra_len) {
  double Hij = 0.00;

  int type[2] = {0};
  diff_type_cuda(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    Hij = get_Hii_cuda(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    Hij = get_HijS_cuda(bra, ket, h1e, h2e, sorb, bra_len);
  } else if (type[0] == 2 && type[1] == 2) {
    Hij = get_HijD_cuda(bra, ket, h1e, h2e, sorb, bra_len);
  }
  return Hij;
}

} // namespace squant

__global__ void tensor_to_onv_kernel(const uint8_t *bra, uint8_t *states,
                                     const int sorb, const int bra_len,
                                     const int m, const size_t n) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= n)
    return;
  int idx = idn % m, idy = idn / m;
  for (int i = 0; i < 8 && idx * 8 + i < sorb; i++) {
    uint8_t value = reinterpret_cast<uint8_t>(bra[sorb * idy + idx * 8 + i]);
    states[idy * 8 * bra_len + idx] |= (value << i);
  }
}

// tensor(uint8): [1, 1, 0, 0] -> onv(uint8): 0b0011, 1 occupied, 0 unoccupied
__host__ void squant::tensor_to_onv_cuda(uint8_t *states, const uint8_t *bra,
                                         const int sorb, const int nbatch,
                                         const int bra_len,
                                         const int tensor_len) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((tensor_len * nbatch + blockDim.x - 1) / blockDim.x);
  tensor_to_onv_kernel<<<gridDim, blockDim>>>(bra, states, sorb, bra_len,
                                              tensor_len, tensor_len * nbatch);
}

__global__ void onv_to_tensor_kernel(double *comb, const unsigned long *bra,
                                     const int sorb, const int bra_len,
                                     const size_t m) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= m)
    return;
  size_t idm = idn / sorb;
  // FIXME: the cost of launching thread. thread > 2**31
  squant::get_zvec_cuda(&bra[idm * bra_len], &comb[idm * sorb], sorb, bra_len,
                        idn % sorb);
}

// onv(unsinged long): 0b0011 -> tensor(double): [1.0. 1.0, 0.0, 0.0],
// 1: occupied, 0: unoccupied
__host__ void squant::onv_to_tensor_cuda(double *comb, const unsigned long *bra,
                                         const int sorb, const int bra_len,
                                         const int nbatch, const size_t numel) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((numel + blockDim.x - 1) / blockDim.x);
  // FIXME: numel > 2*
  onv_to_tensor_kernel<<<gridDim, blockDim>>>(comb, bra, sorb, bra_len, numel);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_Hij_kernel_2D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t bra_len,
                                  const int n, const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idm * bra_len], h1e, h2e, sorb, nele, bra_len);
}

__global__ void get_Hij_kernel_3D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t bra_len,
                                  const int n, const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idn * m * bra_len + idm * bra_len], h1e, h2e,
      sorb, nele, bra_len);
}

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv)
// local energy -> (nbatch, ncomb)
__host__ void squant::get_Hij_3D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int bra_len, const int nbatch,
                                      const int ncomb) {
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_Hij_kernel_3D<<<gridDim, blockDim>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                           bra_len, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

// <i|H|j> matrix, i,j: 2D (nbatch, onv)
// construct Hij matrix -> (nbatch1, nbatch2)
__host__ void squant::get_Hij_2D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int bra_len, const int n,
                                      const int m) {
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (m + blockDim.y - 1) / blockDim.y);
  get_Hij_kernel_2D<<<gridDim, blockDim>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                           bra_len, n, m);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_merged_ovlst_kernel(const unsigned long *bra, int *merged,
                                        int sorb, int nele, int bra_len,
                                        int n) {
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= n)
    return;
  squant::get_ovlst_cuda(&bra[idm * bra_len], &merged[idm * sorb], sorb, nele,
                         bra_len);
}

__host__ void squant::get_merged_cuda(const unsigned long *bra, int *merged,
                                      const int sorb, const int nele,
                                      const int bra_len, const int nbatch) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  get_merged_ovlst_kernel<<<gridDim, blockDim>>>(bra, merged, sorb, nele,
                                                 bra_len, nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_comb_SD_kernel(unsigned long *comb, double *comb_bit,
                                   const int *merged, int sorb, int bra_len,
                                   int noA, int noB, int nbatch, int ncomb) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbatch || idm >= ncomb || idm == 0)
    return;
  // comb[idn, idm], merged[idn], comb_bit[idn, idm]
  squant::get_comb_SD_cuda(&comb[idn * ncomb * bra_len + idm * bra_len],
                           &comb_bit[idn * ncomb * sorb + idm * sorb],
                           &merged[idn * sorb], idm - 1, sorb, noA, noB);
}

__global__ void get_comb_SD_kernel(unsigned long *comb, const int *merged,
                                   const int sorb, const int bra_len,
                                   const int noA, const int noB,
                                   const int nbatch, const int ncomb) {
  const int idn = blockIdx.x * blockDim.x + threadIdx.x;
  const int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbatch || idm >= ncomb || idm == 0)
    return;
  // comb[idn, idm], merged[idn]
  squant::get_comb_SD_cuda(&comb[idn * ncomb * bra_len + idm * bra_len],
                           &merged[idn * sorb], idm - 1, sorb, noA, noB);
}

// get all Singles-Doubles for given onv(2D)
__host__ void squant::get_comb_cuda(unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  // comb: (nbatch, ncomb, bra_len) merged: (nbatch, sorb)
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_comb_SD_kernel<<<gridDim, blockDim>>>(comb, merged_ovlst, sorb, bra_len,
                                            noA, noB, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

// get all Singles-Doubles and states(3D: nbatch, ncomb, sorb) for given onv(2D)
__host__ void squant::get_comb_cuda(double *comb_bit, unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  // comb: (nbatch, ncomb, bra_len), comb_bit: (nbatch, ncomb, sorb)
  // merged: [nbatch, sorb]
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_comb_SD_kernel<<<gridDim, blockDim>>>(comb, comb_bit, merged_ovlst, sorb,
                                            bra_len, noA, noB, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void permuate_sgn_kernel(const int64_t *image2,
                                    const int64_t *onstate, int64_t *index,
                                    int64_t *sgn, const int size,
                                    const size_t nbatch) {
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= nbatch)
    return;
  sgn[idm] = squant::permute_sgn_cuda(image2, &onstate[idm * size],
                                      &index[idm * size], size);
}

__host__ void squant::permute_sng_batch_cuda(const int64_t *image2,
                                             const int64_t *onstate,
                                             int64_t *index, int64_t *sgn,
                                             const int size,
                                             const int64_t nbatch) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  permuate_sgn_kernel<<<gridDim, blockDim>>>(image2, onstate, index, sgn, size,
                                             nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void array_index_kernel(double *data_ptr, int64_t *index,
                                   int64_t length, int64_t offset,
                                   double **ptr_array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  ptr_array[idn] = data_ptr + index[idn] + offset;
  // printf("idn: %ld, index %ld, ptr_array: %p\n", idn, index[idn], (void
  // *)ptr_array[idn]);
}

__host__ void array_index_cuda(double *data_ptr, int64_t *index, int64_t length,
                               int64_t offset, double **ptr_array) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  array_index_kernel<<<gridDim, blockDim>>>(data_ptr, index, length, offset,
                                            ptr_array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_array_kernel(double *data_ptr, int64_t *index,
                                 int64_t length, int64_t offset,
                                 double *array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  array[idn] = *(data_ptr + index[idn] + offset);
}

__host__ void get_array_cuda(double *data_ptr, int64_t *index, int64_t length,
                             int64_t offset, double *array) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  get_array_kernel<<<gridDim, blockDim>>>(data_ptr, index, length, offset,
                                          array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void ones_array_kernel(double *data_ptr, int64_t length,
                                  int64_t stride, int64_t offset) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  data_ptr[idn * stride + offset] = 1.0;
}

__host__ void ones_array_cuda(double *data_ptr, int64_t length, int64_t stride,
                              int64_t offset) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  ones_array_kernel<<<gridDim, blockDim>>>(data_ptr, length, stride, offset);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void print_ptr_ptr_kernel(double *data_ptr, double **ptr_array,
                                     int64_t *dr, int64_t *dc, size_t nbatch) {
  for (size_t i = 0; i < nbatch; i++) {
    printf("%ld-th:(%li, %li) \n", i, dr[i], dc[i]);
    for (size_t j = 0; j < dr[i]; j++) {
      for (size_t k = 0; k < dc[i]; k++) {
        printf("%f ", ptr_array[i][j * dc[i] + k]);
      }
      printf(" \n");
    }
  }
}

__global__ void print_ptr_ptr_kernel(double *data_ptr, double **ptr_array,
                                     int64_t *dr, size_t nbatch) {
  for (size_t i = 0; i < nbatch; i++) {
    printf("%ld-th:%li \n", i, dr[i]);
    for (size_t j = 0; j < dr[i]; j++) {
      printf("%f ", ptr_array[i][j]);
    }
    printf(" \n");
  }
}

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, int64_t *dc, size_t nbatch) {
  print_ptr_ptr_kernel<<<1, 1>>>(data_ptr, ptr_array, dr, dc, nbatch);
}

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, size_t nbatch) {
  print_ptr_ptr_kernel<<<1, 1>>>(data_ptr, ptr_array, dr, nbatch);
}

__global__ void swap_pointers_kernel(double **ptr_ptr, double **ptr_ptr_1) {
  double **tmp = ptr_ptr_1;
  tmp = ptr_ptr_1;
  ptr_ptr_1 = ptr_ptr;
  ptr_ptr = tmp;
  tmp = nullptr;
}

__host__ void swap_pointers_cuda(double **ptr_ptr, double **ptr_ptr_1) {
  swap_pointers_kernel<<<1, 1>>>(ptr_ptr, ptr_ptr_1);
}

__global__ void convert_sites_kernel(
    const int64_t *onstate, const int nphysical, const int64_t *data_index,
    const int64_t *qrow_qcol, const int64_t *qrow_qcol_index,
    const int64_t *qrow_qcol_shape, const int64_t *ista,
    const int64_t *ista_index, const int64_t *image2, const int64_t nbatch,
    int64_t *data_info, bool *sym_array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= nbatch)
    return;
  // onstate: [nbatch, nphysical * 2]
  // data_info [nbatch, nphysical, 3]
  squant::sites_sym_index(&onstate[idn * nphysical * 2], nphysical, data_index,
                          qrow_qcol, qrow_qcol_index, qrow_qcol_shape, ista,
                          ista_index, image2, nbatch, &data_info[idn],
                          &sym_array[idn]);
}

__host__ void convert_sites_cuda(const int64_t *onstate, const int nphysical,
                                 const int64_t *data_index,
                                 const int64_t *qrow_qcol,
                                 const int64_t *qrow_qcol_index,
                                 const int64_t *qrow_qcol_shape,
                                 const int64_t *ista, const int64_t *ista_index,
                                 const int64_t *image2, const int64_t nbatch,
                                 int64_t *data_info, bool *sym_array) {
  // XXX: how to allocate blockDim???, register overflow if blockDim =
  // MAX_THREAD
  dim3 blockDim(256);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  convert_sites_kernel<<<gridDim, blockDim>>>(
      onstate, nphysical, data_index, qrow_qcol, qrow_qcol_index,
      qrow_qcol_shape, ista, ista_index, image2, nbatch, data_info, sym_array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void merge_idx_kernel(int64_t *merge_counts, const int64_t *idx,
                                 const int64_t *counts, const int64_t batch) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= batch)
    return;
  // Notice: atomicadd
  merge_counts[idx[idn]] += counts[idn];
}

__host__ void merge_idx_cuda(int64_t *merge_counts, const int64_t *idx,
                             const int64_t *counts, const int64_t batch) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((batch + blockDim.x - 1) / blockDim.x);
  merge_idx_kernel<<<gridDim, blockDim>>>(merge_counts, idx, counts, batch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();
}

__device__ void constrain_charts(const int64_t key, double *result) {
  // {10, 6, 14, 9, 5, 13, 11, 7, 15};
  if (key == 10) {
    result[0] = 1;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 6) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 1;
    result[3] = 0;
  } else if (key == 14) {
    result[0] = 1;
    result[1] = 0;
    result[2] = 1;
    result[3] = 0;
  } else if (key == 9) {
    result[0] = 0;
    result[1] = 1;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 5) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 1;
  } else if (key == 13) {
    result[0] = 0;
    result[1] = 1;
    result[2] = 0;
    result[3] = 1;
  } else if (key == 11) {
    result[0] = 1;
    result[1] = 1;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 7) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 1;
    result[3] = 1;
  } else if (key == 15) {
    result[0] = 1;
    result[1] = 1;
    result[2] = 1;
    result[3] = 1;
  }
}

__global__ void constrain_lookup_table_kernel(const int64_t *sym_index,
                                              double *result,
                                              const int64_t nbatch) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= nbatch)
    return;
  constrain_charts(sym_index[idn], &result[idn * 4]);
}

__host__ void constrain_lookup_table(const int64_t *sym_index, double *result,
                                     const int64_t nbatch) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  constrain_lookup_table_kernel<<<gridDim, blockDim>>>(sym_index, result,
                                                       nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();
}

template <typename IntType>
__device__ int64_t BigInteger_device(const IntType *arr, const IntType *target,
                                     const int64_t arr_length,
                                     const int64_t target_length = 1,
                                     bool little_endian = true) {
  // arr: [arr_length, targe_length] 2D array but arr is point not point-point
  // arr is array of the great uint64 or others [12, 13] => 2**64 + 12
  // target: [targe_length]
  // little_endian: [12, 13] => 13 * 2**64 + 12
  // big_endian: [12, 13] => 12 * 2**64 + 12
  int64_t left = 0;
  int64_t right = arr_length - 1;

  auto compare = [&arr, &target, target_length,
                  little_endian](const IntType *mid_element) -> int {
    if (little_endian) {
      for (int64_t i = target_length - 1; i >= 0; i--) {
        if (mid_element[i] < target[i]) {
          return -1;
        } else if (mid_element[i] > target[i]) {
          return 1;
        }
      }
    } else {
      for (int64_t i = 0; i < target_length; i--) {
        if (mid_element[i] < target[i]) {
          return -1;
        } else if (mid_element[i] > target[i]) {
          return 1;
        }
      }
    }
    return 0;
  };

  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    int64_t mid_index = mid * target_length;
    const IntType *mid_element = &arr[mid_index];
    int result = compare(mid_element);

    if (result == 0) {
      return mid;
    } else if (result < 0) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return -1;
}

__global__ void BigInteger_kernel(const unsigned long *arr,
                                  const unsigned long *target, int64_t *result,
                                  const int64_t nbatch,
                                  const int64_t arr_length,
                                  const int64_t target_length,
                                  bool little_endian) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= nbatch)
    return;
  result[idn] = BigInteger_device<unsigned long>(
      arr, &target[idn * target_length], arr_length, target_length,
      little_endian);
};

__host__ void binary_search_BigInteger_cuda(
    const unsigned long *arr, const unsigned long *target, int64_t *result,
    const int64_t nbatch, const int64_t arr_length,
    const int64_t target_length = 1, bool little_endian = true) {
  dim3 blockDim(MAX_THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  BigInteger_kernel<<<gridDim, blockDim>>>(
      arr, target, result, nbatch, arr_length, target_length, little_endian);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();
}