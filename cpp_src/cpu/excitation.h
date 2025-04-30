#pragma once
#include "../common/utils.h"

namespace squant {

inline void unpack_canon(const int ij, int *s) {
  int i = std::sqrt((ij + 1) * 2) + 0.5;
  int j = ij - i * (i - 1) / 2;
  s[0] = i;
  s[1] = j;
}

int get_Num_SinglesDoubles(const int sorb, const int noA, const int noB);

void unpack_SinglesDoubles(const int sorb, const int noA, const int noB,
                           const int idx, int *idx_lst);

void get_comb_SD(unsigned long *comb, const int *merged, const int r0,
                 const int sorb, const int len, const int noA, const int noB);

void get_comb_SD(unsigned long *comb, double *lst, const int *merged,
                 const int r0, const int sorb, const int len, const int noA,
                 const int noB);

template <typename T>
T get_comb_SD_fused(unsigned long *comb, const int *merged, const T *h1e,
                    const T *h2e, unsigned long *bra, const int r0,
                    const int sorb, const int len, const int noA,
                    const int noB);

}  // namespace squant