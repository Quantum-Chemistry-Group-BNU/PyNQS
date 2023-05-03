#pragma once
#include "../common/utils.h"

namespace squant {
void unpack_canon(int ij, int *s);

int get_Num_SinglesDoubles(int sorb, int noA, int noB);

void unpack_SinglesDoubles(int sorb, int noA, int noB, int idx, int *idx_lst);

void get_comb_SD(unsigned long *bra, int merged, int r0, int n, int len, int noa, int nob);

void get_comb_SD(unsigned long *bra, double *lst,int merged, int r0, int n, int len, int noa, int nob);

} // namespace