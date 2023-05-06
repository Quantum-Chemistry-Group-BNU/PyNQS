#include "onstate_cuda.h"

namespace squant {

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

__device__ int parity_cuda(const unsigned long *bra, const int n) {
  int p = 0;
  for (int i = 0; i < n / 64; i++) {
    p ^= get_parity_cuda(bra[i]);
  }
  if (n % 64 != 0) {
    p ^= get_parity_cuda((bra[n / 64] & get_ones_cuda(n % 64)));
  }
  return -2 * p + 1;
}

__device__ void diff_orb(const unsigned long *bra, const unsigned long *ket,
                         const int _len, int *cre, int *ann) {
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

__device__ void get_olst_ab(const unsigned long *bra, int *olst,
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

__device__ void get_vlst_cuda(const unsigned long *bra, int *vlst, const int n,
                              const int _len) {
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cuda(n % 64));
    while (tmp != 0) {
      int j = __ctzl(tmp);
      vlst[ic] = i * 64 + j;
      ic++;
      tmp &= ~(1ULL << j);
    }
  }
}

__device__ void get_vlst_ab_cuda(const unsigned long *bra, int *vlst,
                                 const int n, const int _len) {
  int ic = 0;
  int idb = 0;
  int ida = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cuda(n % 64));
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
  get_olst_cuda(bra, merged, bra_len);
  get_vlst_cuda(bra, merged + nele, sorb, bra_len);
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

} // namespace squant