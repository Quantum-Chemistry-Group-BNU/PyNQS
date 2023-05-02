#pragma once
#include "../common/utils.h"

NAMESPACE_BEGIN(squant)

inline int popcnt_cpu(const unsigned long x) { return __builtin_popcountl(x); }

inline int get_parity_cpu(const unsigned long x) {
  return __builtin_parityl(x);
}

inline unsigned long get_ones_cpu(const int n) { return (1ULL << n) - 1ULL; }

inline double num_parity_cpu(unsigned long x, int i) {
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

void diff_type_cpu(unsigned long *bra, unsigned long *ket, int *p, int _len);

int parity_cpu(unsigned long *bra, int n);

void get_olst_cpu(unsigned long *bra, int *olst, int _len);

// olst: abab
void get_olst_cpu_ab(unsigned long *bra, int *olst, int _len);

void get_vlst_cpu(unsigned long *bra, int *vlst, int n, int _len);

// vlst: abab
void get_vlst_cpu_ab(unsigned long *bra, int *vlst, int n, int _len);

void diff_orb_cpu(unsigned long *bra, unsigned long *ket, int _len, int *cre,
                  int *ann);

NAMESPACE_END(fock)