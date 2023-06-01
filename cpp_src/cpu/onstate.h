#pragma once
#include "../common/utils.h"

namespace squant {

inline int popcnt_cpu(unsigned long x) { return __builtin_popcountl(x); }

inline int get_parity_cpu(unsigned long x) { return __builtin_parityl(x); }

inline unsigned long get_ones_cpu(unsigned long n) {
  return (1ULL << n) - 1ULL;
}

inline double num_parity_cpu(unsigned long x, int i) {
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

void diff_type_cpu(const unsigned long *bra, const unsigned long *ket, int *p,
                   const int _len);

int parity_cpu(const unsigned long *bra, const int sorb);

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

void get_zvec_cpu(const unsigned long *bra, double *lst, const int sorb,
                  const int bra_len);

}  // namespace squant