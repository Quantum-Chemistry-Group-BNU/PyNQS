#include "onstate.h"
#include "utils.h"

namespace squant {

void diff_type_cpu(unsigned long *bra, unsigned long *ket, int *p, int _len) {
  unsigned long idiff, icre, iann;
  for (int i = _len - 1; i >= 0; i--) {
    idiff = bra[i] ^ ket[i];
    icre = idiff & bra[i];
    iann = idiff & ket[i];
    p[0] += popcnt_cpu(icre);
    p[1] += popcnt_cpu(iann);
  }
}

int parity_cpu(unsigned long *bra, int n) {
  int p = 0;
  for (int i = 0; i < n / 64; i++) {
    p ^= get_parity_cpu(bra[i]);
  }
  if (n % 64 != 0) {
    p ^= get_parity_cpu((bra[n / 64] & get_ones_cpu(n % 64)));
  }
  return -2 * p + 1;
}

void get_olst_cpu(unsigned long *bra, int *olst, int _len) {
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

void get_olst_cpu_ab(unsigned long *bra, int *olst, int _len) {
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
      if ( s & 1){
        idb++;
        idx = 2 * idb - 1;
      }else{
        ida++;
        idx = 2 * (ida -1);
      }
      olst[idx] = s;
      tmp &= ~(1ULL << j);
    }
  }
}

void get_vlst_cpu(unsigned long *bra, int *vlst, int n, int _len) {
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cpu(n % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      vlst[ic] = i * 64 + j;
      ic++;
      tmp &= ~(1ULL << j);
    }
  }
}

void get_vlst_cpu_ab(unsigned long *bra, int *vlst, int n, int _len) {
  int ic = 0;
  int idb = 0;
  int ida = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cpu(n % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      int s = i * 64 + j;
      if (s & 1){
        idb++;
        ic = 2 * idb - 1;
      }else{
        ida++;
        ic = 2 * (ida - 1);
      }
      vlst[ic] = s;
      tmp &= ~(1ULL << j);
    }
  }
}

void diff_orb_cpu(unsigned long *bra, unsigned long *ket, int _len, int *cre,
                  int *ann) {
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

} // namespace squant