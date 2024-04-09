#include "onstate_cuda.h"
#include <cstddef>
#include <cstdio>

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
  // get_olst_ab_cuda(bra, merged, bra_len);
  // get_vlst_ab_cuda(bra, merged + nele, sorb, bra_len);
  // occupied orbital(abab) -> virtual orbital(abab), notice: alpha != beta
  // e.g. 0b00011100 -> 23410567
  unsigned long tmp;
  int idx = 0;
  int ida = 0;
  int idb = 0;
  // occupied orbital
  for (int i = 0; i < bra_len; i++) {
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
      merged[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
  // virtual orbital
  for (int i = 0; i < bra_len; i++) {
    tmp =
        (i != bra_len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cuda(sorb % 64));
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
      merged[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
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

} // namespace squant