#pragma once
#include "../common/utils.h"

namespace squant {

inline int popcnt_cpu(unsigned long x) { return __builtin_popcountl(x); }

inline int get_parity_cpu(unsigned long x) { return __builtin_parityl(x); }

inline unsigned long get_ones_cpu(const int n) {
  return n == 64 ? ~0ULL : (1ULL << n) - 1ULL;
}

template<typename T= double>
inline T num_parity_cpu(unsigned long x, int i) {
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

void diff_type_cpu(const unsigned long *bra, const unsigned long *ket, int *p,
                   const int _len);

int parity_cpu(const unsigned long *bra, const int n);

void diff_orb_cpu(const unsigned long *bra, const unsigned long *ket,
                  const int _len, int *cre, int *ann);

void get_olst_cpu(const unsigned long *bra, int *olst, const int _len);

void get_vlst_cpu(const unsigned long *bra, int *vlst, const int sorb,
                  const int _len);

// vlst: abab
void get_vlst_ab_cpu(const unsigned long *bra, int *vlst, const int sorb,
                     const int _len);

// olst: abab
void get_olst_ab_cpu(const unsigned long *bra, int *olst, const int _len);

void get_olst_vlst_ab_cpu(const unsigned long *bra, int *lst, const int sorb,
                          const int _len);

// void get_zvec_cpu(const unsigned long *bra, double *lst, const int sorb,
//                   const int bra_len);
template<typename T>
void get_zvec_cpu(const unsigned long *bra, T *lst, const int sorb,
                  const int bra_len) {
  constexpr int block = 64;
  int idx = 0;
  for (int i = 0; i < bra_len - 1; i++) {
    for (int j = 1; j <= block; j++) {
      lst[idx] = num_parity_cpu<T>(bra[i], j);
      idx++;
    }
  }
  int reset = sorb % block;
  reset = reset > 0 ? reset : 64;
  for (int j = 1; j <= reset; j++) {
    // if (idx >= sorb) break;
    lst[idx] = num_parity_cpu<T>(bra[bra_len - 1], j);
    idx++;
  }
  assert(idx == sorb);
}

int64_t permute_sgn_cpu(const int64_t *image2, const int64_t *onstate,
                        const int size);
}  // namespace squant