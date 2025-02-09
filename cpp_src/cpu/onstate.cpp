#include "onstate.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
// #include "utils.h"

namespace squant {

void diff_type_cpu(const unsigned long *bra, const unsigned long *ket, int *p,
                   const int _len) {
  unsigned long idiff, icre, iann;
  for (int i = _len - 1; i >= 0; i--) {
    idiff = bra[i] ^ ket[i];
    icre = idiff & bra[i];
    iann = idiff & ket[i];
    p[0] += popcnt_cpu(icre);
    p[1] += popcnt_cpu(iann);
  }
}

int parity_cpu(const unsigned long *bra, const int n) {
  int p = 0;
  for (int i = 0; i < n / 64; i++) {
    p ^= get_parity_cpu(bra[i]);
  }
  if (n % 64 != 0) {
    // TODO: check
    p ^= get_parity_cpu((bra[n / 64] & get_ones_cpu(n % 64)));
  }
  return -2 * p + 1;
}

void diff_orb_cpu(const unsigned long *bra, const unsigned long *ket,
                  const int _len, int *cre, int *ann) {
  int idx_cre = 0;
  int idx_ann = 0;
  for (int i = _len - 1; i >= 0; i--) {
    unsigned long idiff = bra[i] ^ ket[i];
    unsigned long icre = idiff & bra[i];
    unsigned long iann = idiff & ket[i];
    while (icre != 0) {
      int j = 63 - __builtin_clzl(icre);  // unsigned long
      cre[idx_cre] = i * 64 + j;
      icre &= ~(1ULL << j);
      idx_cre++;
    }
    while (iann != 0) {
      int j = 63 - __builtin_clzl(iann);  // unsigned long
      ann[idx_ann] = i * 64 + j;
      iann &= ~(1ULL << j);
      idx_ann++;
    }
  }
}

void get_olst_cpu(const unsigned long *bra, int *olst, const int _len) {
  unsigned long tmp;
  int idx = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      olst[idx] = i * 64 + j;
      tmp &= ~(1ULL << j);
      idx++;
    }
  }
}

void get_olst_ab_cpu(const unsigned long *bra, int *olst, const int _len) {
  // abab
  unsigned long tmp;
  int idx = 0;
  int ida = 0;
  int idb = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
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

void get_vlst_cpu(const unsigned long *bra, int *vlst, const int sorb,
                  const int _len) {
  // int ic = 0;
  // auto onv = std::bitset<64>(bra[0]);
  // for (auto i = 0; i < sorb; i++) {
  //   if (onv[i] == 0) {
  //     vlst[ic] = i;
  //     ic++;
  //   }
  // }
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1)
              ? (~bra[i])
              : ((~bra[i]) & get_ones_cpu(sorb % 64 == 0 ? 64 : sorb % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      vlst[ic] = i * 64 + j;
      ic++;
      tmp &= ~(1ULL << j);
    }
  }
}

void get_vlst_ab_cpu(const unsigned long *bra, int *vlst, const int sorb,
                     const int _len) {
  int ic = 0;
  int idb = 0;
  int ida = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    tmp = (i != _len - 1)
              ? (~bra[i])
              : ((~bra[i]) & get_ones_cpu(sorb % 64 == 0 ? 64 : sorb % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
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

void get_olst_vlst_ab_cpu(const unsigned long *bra, int *lst, const int sorb,
                          const int _len) {
  // std::bitset<sorb> bitset(bra[0]);
  // std::cout << "bra " << bitset << std::endl;
  // occupied orbital(abab) -> virtual orbital(abab), notice: alpha != beta
  // e.g. 0b00011100 -> 23410567
  unsigned long tmp;
  int idx = 0;
  int ida = 0;
  int idb = 0;
  // occupied orbital
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      int s = i * 64 + j;
      if (s & 1) {
        idb++;
        idx = 2 * idb - 1;
      } else {
        ida++;
        idx = 2 * (ida - 1);
      }
      lst[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
  // virtual orbital
  for (int i = 0; i < _len; i++) {
    tmp = (i != _len - 1)
              ? (~bra[i])
              : ((~bra[i]) & get_ones_cpu(sorb % 64 == 0 ? 64 : sorb % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      int s = i * 64 + j;
      if (s & 1) {
        idb++;
        idx = 2 * idb - 1;
      } else {
        ida++;
        idx = 2 * (ida - 1);
      }
      lst[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
}

void get_zvec_cpu(const unsigned long *bra, double *lst, const int sorb,
                  const int bra_len) {
  constexpr int block = 64;
  int idx = 0;
  for (int i = 0; i < bra_len - 1; i++) {
    for (int j = 1; j <= block; j++) {
      lst[idx] = num_parity_cpu(bra[i], j);
      idx++;
    }
  }
  int reset = sorb % block;
  reset = reset > 0 ? reset : 64;
  for (int j = 1; j <= reset; j++) {
    // if (idx >= sorb) break;
    lst[idx] = num_parity_cpu(bra[bra_len - 1], j);
    idx++;
  }
  assert(idx == sorb);
}

int64_t permute_sgn_cpu(const int64_t *image2, const int64_t *onstate,
                        const int size) {
  std::vector<int64_t> index(size);
  std::iota(index.begin(), index.end(), 0);
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

}  // namespace squant