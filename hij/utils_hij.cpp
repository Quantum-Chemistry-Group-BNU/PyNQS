#include "utils_hij.h"
#include <bitset>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <algorithm>


#include "ATen/core/TensorBody.h"

inline int popcnt_cpu(const unsigned long x) { return __builtin_popcountl(x); }
inline int get_parity_cpu(const unsigned long x) {
  return __builtin_parityl(x);
}

inline unsigned long get_ones_cpu(const int n) {
  return (1ULL << n) - 1ULL;
}  // parenthesis must be added due to priority
inline double num_parity_cpu(unsigned long x, int i) {
  return (x >> (i - 1) & 1) ? 1.00 : -1.00;
}

std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

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

void get_olst_cpu(unsigned long *bra, int *olst, int *olst_a, int *olst_b,
                  int _len) {
  unsigned long tmp;
  int ida = 0;
  int idb = 0;
  int idx = 0;
  for (int i = 0; i < _len; i++) {
    tmp = bra[i];
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      int s = i * 64 + j;
      olst[idx] = s;
      idx++;
      if (s & 1) {
        olst_b[idb] = s;
        idb++;
      } else {
        olst_a[ida] = s;
        ida++;
      }
      tmp &= ~(1ULL << j);
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

void get_vlst_cpu(unsigned long *bra, int *vlst, int *vlst_a, int *vlst_b,
                  int n, int _len) {
  int ida = 0;
  int idb = 0;
  int ic = 0;
  unsigned long tmp;
  for (int i = 0; i < _len; i++) {
    // be careful about the virtual orbital case
    tmp = (i != _len - 1) ? (~bra[i]) : ((~bra[i]) & get_ones_cpu(n % 64));
    while (tmp != 0) {
      int j = __builtin_ctzl(tmp);
      int s = i * 64 + j;
      vlst[ic] = s;
      ic++;
      if (s & 1) {
        vlst_b[idb] = s;
        idb++;
      } else {
        vlst_a[ida] = s;
        ida++;
      }
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

void get_zvec_cpu(unsigned long *bra, double *lst, const int sorb,
                  const int bra_len) {
  int idx = 0;
  for (int i = 0; i < bra_len; i++) {
    for (int j = 1; j <= 64; j++) {
      if (idx >= sorb) break;
      lst[idx] = num_parity_cpu(bra[i], j);
      idx++;
    }
  }
}

double h1e_get_cpu(double *h1e, size_t i, size_t j, size_t sorb) {
  return h1e[j * sorb + i];
}

double h2e_get_cpu(double *h2e, size_t i, size_t j, size_t k, size_t l) {
  if ((i == j) || (k == l)) return 0.00;
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
    val = sgn * h2e[ijkl];  // sgn * conjugate(h2e[ijkl])
  }
  return val;
}

// TODO: This maybe error, spin multiplicity is not equal for very comb
void get_comb_2d(unsigned long *bra, unsigned long *comb, int n, int len,
                 int no, int nv, bool ms = true) {
  // auto vlst = std::make_unique<int[]>(nv);
  // auto olst = std::make_unique<int[]>(no);
  int vlst[MAX_NV] = {0};
  int olst[MAX_NO] = {0};
  get_olst_cpu(bra, olst, len);
  get_vlst_cpu(bra, vlst, n, len);

  for (int i = 0; i < len; i++) {
    comb[i] = bra[i];
  }
  int idx = 1; /*This is the index of comb*/
  // spin multiplicity(ms) is equal??
  bool flag = false;
  // Singles nv/2 * na * nb
  int idx_singles = 0;
  for (int i = 0; i < no; i++) {
    for (int j = 0; j < nv; j++) {
      if (ms) {
        // olst[i] and vlst[j] is identical spin orbital
        if ((olst[i] & 1) == (vlst[j] & 1)) {
          flag = true;
        } else {
          // flag = false;
          continue;
        }
      }
      if ((not ms) || (ms && flag)) {
        int idi = len * idx + olst[i] / 64;
        int idj = len * idx + vlst[j] / 64;
        comb[idi] = bra[olst[i] / 64];
        comb[idj] = bra[vlst[j] / 64];
        BIT_FLIP(comb[idi], olst[i] % 64);
        BIT_FLIP(comb[idj], vlst[j] % 64);
        idx++;
        flag = false;
        idx_singles++;
        // std::cout << "comb[idj]: " << std::bitset<8> (comb[idj]) <<
        // std::endl;
      }
    }
  }
  // std::cout << "Singles: " << idx_s << std::endl;
  int idx_doubles = 0;
  // Doubles
  for (int i = 0; i < no; i++) {
    for (int j = i + 1; j < no; j++) {
      for (int k = 0; k < nv; k++) {
        for (int l = k + 1; l < nv; l++) {
          if (ms) {
            /***
            Doubles: (ij->kl)
            1. i/k:a, l/k:b ab->ab
            2. i/k:b, l/k:b ba->ba
            3. i/l:a, j/k:b ab->ba
            4. i/l:b, j/k:a ba->ab
            5. i/l:a, j/k:a aa->aa s.t. i < j and k < l
            6. i/l:b, j/k:b bb->bb s.t. i < j and k < l
            Notice: 4, 5 only just for programming convenience.
            ***/
            // 1, 2, 5, 6
            bool flag_one = ((olst[i] & 1) == (vlst[k] & 1) &&
                             ((olst[j] & 1) == (vlst[l] & 1)));
            // 3, 4, 5, 6
            bool flag_two = ((olst[i] & 1) == (vlst[l] & 1) &&
                             ((olst[j] & 1) == (vlst[k] & 1)));
            if (flag_one || flag_two) {
              // std::cout << "flag_one: " << flag_one << " flag_two: " <<
              // flag_two << std::endl;
              flag = true;
            } else {
              continue;
            }
          }
          if ((not ms) || (ms && flag)) {
            int idi = len * idx + olst[i] / 64;
            int idj = len * idx + olst[j] / 64;
            int idk = len * idx + vlst[k] / 64;
            int idl = len * idx + vlst[l] / 64;
            comb[idi] = bra[olst[i] / 64];
            comb[idj] = bra[olst[j] / 64];
            comb[idk] = bra[vlst[k] / 64];
            comb[idl] = bra[vlst[l] / 64];
            BIT_FLIP(comb[idi], olst[i] % 64);
            BIT_FLIP(comb[idj], olst[j] % 64);
            BIT_FLIP(comb[idk], vlst[k] % 64);
            BIT_FLIP(comb[idl], vlst[l] % 64);
            idx++;
            flag = false;
            idx_doubles++;
            // std::cout << "comb[idj]: " << std::bitset<8> (comb[idj]) <<
            // std::endl;
          }
        }
      }
    }
  }
  // std::cout << "Double: " << idx_double << std::endl;
  // std::cout << "Singles: " << idx_singles << std::endl;
}

void get_comb_2d(unsigned long *bra, unsigned long *comb, int n, int len,
                 int noa, int nob, int nva, int nvb) {
  int olst[MAX_NO] = {0};
  int vlst[MAX_NV] = {0};
  int olst_a[MAX_NOA] = {0};
  int olst_b[MAX_NOB] = {0};
  int vlst_a[MAX_NOA] = {0};
  int vlst_b[MAX_NOB] = {0};
  get_olst_cpu(bra, olst, olst_a, olst_b, len);
  get_vlst_cpu(bra, vlst, vlst_a, vlst_b, n, len);

  for (int i = 0; i < len; i++) {
    comb[i] = bra[i];
  }
  int idx = 1;
  int idx_singles = 0;
  // a->a: noa * nva
  for (int i = 0; i < noa; i++) {
    for (int j = 0; j < nva; j++) {
      int idi = len * idx + olst_a[i] / 64;
      int idj = len * idx + vlst_a[j] / 64;
      comb[idi] = bra[olst_a[i] / 64];
      comb[idj] = bra[vlst_a[j] / 64];
      BIT_FLIP(comb[idi], olst_a[i] % 64);
      BIT_FLIP(comb[idj], vlst_a[j] % 64);
      idx++;
      idx_singles += 1;
    }
  }
  // b->b: nob * nvb
  for (int i = 0; i < nob; i++) {
    for (int j = 0; j < nvb; j++) {
      int idi = len * idx + olst_b[i] / 64;
      int idj = len * idx + vlst_b[j] / 64;
      comb[idi] = bra[olst_b[i] / 64];
      comb[idj] = bra[vlst_b[j] / 64];
      BIT_FLIP(comb[idi], olst_b[i] % 64);
      BIT_FLIP(comb[idj], vlst_b[j] % 64);
      idx++;
      idx_singles++;
    }
  }
  // std::cout << "Singles: " << idx_singles << std::endl;
  int idx_doubles = 0;
  // aa->aa, noa * (noa - 1) * nva * (nva - 1) / 4
  for (int i = 0; i < noa; i++) {
    for (int j = i + 1; j < noa; j++) {
      for (int k = 0; k < nva; k++) {
        for (int l = k + 1; l < nva; l++) {
          int idi = len * idx + olst_a[i] / 64;
          int idj = len * idx + olst_a[j] / 64;
          int idk = len * idx + vlst_a[k] / 64;
          int idl = len * idx + vlst_a[l] / 64;
          comb[idi] = bra[olst_a[i] / 64];
          comb[idj] = bra[olst_a[j] / 64];
          comb[idk] = bra[vlst_a[k] / 64];
          comb[idl] = bra[vlst_a[l] / 64];
          BIT_FLIP(comb[idi], olst_a[i] % 64);
          BIT_FLIP(comb[idj], olst_a[j] % 64);
          BIT_FLIP(comb[idk], vlst_a[k] % 64);
          BIT_FLIP(comb[idl], vlst_a[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
  // bb->bb: nob * (nob - 1) * nvb * (nvb - 1) / 4
  for (int i = 0; i < nob; i++) {
    for (int j = i + 1; j < nob; j++) {
      for (int k = 0; k < nvb; k++) {
        for (int l = k + 1; l < nvb; l++) {
          int idi = len * idx + olst_b[i] / 64;
          int idj = len * idx + olst_b[j] / 64;
          int idk = len * idx + vlst_b[k] / 64;
          int idl = len * idx + vlst_b[l] / 64;
          comb[idi] = bra[olst_b[i] / 64];
          comb[idj] = bra[olst_b[j] / 64];
          comb[idk] = bra[vlst_b[k] / 64];
          comb[idl] = bra[vlst_b[l] / 64];
          BIT_FLIP(comb[idi], olst_b[i] % 64);
          BIT_FLIP(comb[idj], olst_b[j] % 64);
          BIT_FLIP(comb[idk], vlst_b[k] % 64);
          BIT_FLIP(comb[idl], vlst_b[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
  // std::cout << "aa-aa/bb-bb : " << idx_doubles << std::endl;
  // ab->ab (noa * nva * nob * nvb)
  for (int i = 0; i < noa; i++) {
    for (int j = 0; j < nob; j++) {
      for (int k = 0; k < nva; k++) {
        for (int l = 0; l < nvb; l++) {
          int idi = len * idx + olst_a[i] / 64;
          int idj = len * idx + olst_b[j] / 64;
          int idk = len * idx + vlst_a[k] / 64;
          int idl = len * idx + vlst_b[l] / 64;
          comb[idi] = bra[olst_a[i] / 64];
          comb[idj] = bra[olst_b[j] / 64];
          comb[idk] = bra[vlst_a[k] / 64];
          comb[idl] = bra[vlst_b[l] / 64];
          BIT_FLIP(comb[idi], olst_a[i] % 64);
          BIT_FLIP(comb[idj], olst_b[j] % 64);
          BIT_FLIP(comb[idk], vlst_a[k] % 64);
          BIT_FLIP(comb[idl], vlst_b[l] % 64);
          idx++;
          idx_doubles++;
        }
      }
    }
  }
}

void get_comb_2d(unsigned long *comb, int *merged, int r0, int n, int len, int noa,
                 int nob) {
  int idx_lst[4] = {0};
  std::cout << "i j k l: ";
  unpack_Singles_Doubles(n, noa, nob, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]] ;
    BIT_FLIP(comb[idx / 64], idx % 64);
  }
  std::cout << std::endl;
}

void get_comb_2d(unsigned long *comb, double *lst, int *merged, int r0, int n, int len,
                 int noa, int nob) {
  int idx_lst[4] = {0};
  unpack_Singles_Doubles(n, noa, nob, r0, idx_lst);
  for (int i = 0; i < 4; i++) {
    int idx = merged[idx_lst[i]];
    BIT_FLIP(comb[idx / 64], idx % 64);
    lst[idx] *= -1;
  }
}

int get_Num_SinglesDoubles(int sorb, int noA, int noB){
  int k = sorb / 2;
  int nvA = k - noA, nvB = k - noB;
  int nSa = noA * nvA, nSb = noB * nvB;
  int nDaa = noA * (noA-1) * nvA * (nvA -1)/4;
  int nDbb = noB * (noB-1) * nvB * (nvB -1)/4;
  int nDab = noA * noB * nvA * nvB;
  return nSa + nSb + nDaa + nDbb + nDab;
}

void unpack_canon(int ij, int *s){
  int i = std::sqrt((ij+ 1) * 2) + 0.5;
  int j = ij - i*(i-1)/2;
  s[0] = i;
  s[1] = j;
}

void unpack_Singles_Doubles(int sorb, int noA, int noB, int idx, int *idx_lst) {
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
      a = 2 * (jdx / noA + noA);// alpha-even; beta-odd
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
      unpack_canon(ijA, s1);
      unpack_canon(abA, s2);
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
      unpack_canon(ijB, s1);
      unpack_canon(abB, s2);
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
  #if 0
  std::cout << "idx: " << idx << " ";
  std::cout << "icase: " << icase << " ";
  std::cout << " i a j b:" << i << " " << a << " " << j << " " << b
            << std::endl;
  #endif 
  idx_lst[0] = i;
  idx_lst[1] = a;
  idx_lst[2] = j;
  idx_lst[3] = b;
}

double get_Hii_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, int sorb, const int nele, int bra_len) {
  double Hii = 0.00;
  // int olst[nele] ={0};
  // int *olst = new int[nele];
  int olst[MAX_NELE] = {0};
  get_olst_cpu(bra, olst, bra_len);

  for (int i = 0; i < nele; i++) {
    int p = olst[i];  //<p|h|p>
    Hii += h1e_get_cpu(h1e, p, p, sorb);
    for (int j = 0; j < i; j++) {
      int q = olst[j];
      Hii += h2e_get_cpu(h2e, p, q, p, q);  //<pq||pq> Storage not continuous
    }
  }
  // delete []olst;
  return Hii;
}

double get_HijS_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len) {
  double Hij = 0.00;
  int p[1], q[1];
  diff_orb_cpu(bra, ket, bra_len, p, q);
  Hij += h1e_get_cpu(h1e, p[0], q[0], sorb);  // hpq
  for (int i = 0; i < bra_len; i++) {
    unsigned long repr = bra[i];
    while (repr != 0) {
      int j = 63 - __builtin_clzl(repr);
      int k = 64 * i + j;
      Hij += h2e_get_cpu(h2e, p[0], k, q[0], k);  //<pk||qk>
      repr &= ~(1ULL << j);
    }
  }
  int sgn = parity_cpu(bra, p[0]) * parity_cpu(ket, q[0]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

double get_HijD_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len) {
  int p[2], q[2];
  diff_orb_cpu(bra, ket, bra_len, p, q);
  int sgn = parity_cpu(bra, p[0]) * parity_cpu(bra, p[1]) *
            parity_cpu(ket, q[0]) * parity_cpu(ket, q[1]);
  double Hij = h2e_get_cpu(h2e, p[0], p[1], q[0], q[1]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

torch::Tensor get_merged_olst_vlst(torch::Tensor bra, const int nele, const int sorb, const int noA, const int noB){
  const int dim = bra.dim();
  const int bra_len = (sorb - 1)/64 + 1;
  int n = 0;
  if (dim == 1){
    n = 1;
  }else{
    n = bra.size(0);
  }
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(bra.layout())
                     .device(bra.device())
                     .requires_grad(false);
  torch::Tensor merged = torch::ones({n, sorb}, options);
  int *merged_ptr = merged.data_ptr<int32_t>();
  for(int i =0; i< n; i++){
    unsigned long *bra_ptr = reinterpret_cast<unsigned long *>(bra[i].data_ptr<uint8_t>());
   get_olst_cpu_ab(bra_ptr, &merged_ptr[i * sorb], bra_len);
   get_vlst_cpu_ab(bra_ptr, &merged_ptr[i * sorb + nele], sorb, bra_len);
  }
  return merged;
}

auto get_nsingles_doubles(const int no, const int nv, bool ms_equal) {
  int nsingles, ndoubles, ncomb;
  // spin multiplicity is equal for very comb
  if (ms_equal) {
    int nvb = nv / 2;
    int nva = nv - nvb;
    int nob = no / 2;
    int noa = no - nob;
    // std::cout << noa << " "<< nob << " " << nva  << " "<< nvb << std::endl;
    nsingles = noa * nva + nob * nvb;
    // std::cout << "nsingles: " << nsingles << std::endl;
    // this is error for radical e.g. H3 H5 ...
    ndoubles = noa * (noa - 1) * nva * (nva - 1) / 4 +
               nob * (nob - 1) * nvb * (nvb - 1) / 4 + noa * nva * nob * nvb;
    // std::cout << "ndoubles: " << ndoubles << std::endl;
  } else {
    nsingles = no * nv;
    ndoubles = no * (no - 1) * nv * (nv - 1) / 4;
  }
  ncomb = 1 + nsingles + ndoubles;
  return std::make_tuple(nsingles, ndoubles, ncomb);
}

double get_Hij_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, size_t sorb, size_t nele, size_t tensor_len,
                   size_t bra_len) {
  /*
  bra/ket: unsigned long
  */
  double Hij = 0.00;
  int type[2] = {0};
  diff_type_cpu(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    Hij = get_Hii_cpu(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    Hij = get_HijS_cpu(bra, ket, h1e, h2e, sorb, bra_len);
    // std::cout << "Singles: " << std::bitset<8>(bra[0]) << " " <<
    // std::bitset<8>(ket[0])  << " "; std::cout << "Hij: " << Hij << std::endl;
  } else if (type[0] == 2 && type[1] == 2) {
    Hij = get_HijD_cpu(bra, ket, h1e, h2e, sorb, bra_len);
    // std::cout << "Double: " << std::bitset<8>(bra[0]) << " " <<
    // std::bitset<8>(ket[0])  << " "; std::cout << "Hij: " << Hij << std::endl;
  }
  return Hij;
}

torch::Tensor get_Hij_mat_cpu(torch::Tensor &bra_tensor,
                              torch::Tensor &ket_tensor,
                              torch::Tensor &h1e_tensor,
                              torch::Tensor &h2e_tensor, const int sorb,
                              const int nele) {
  auto t3 = get_time();
  int n, m;
  const int ket_dim = ket_tensor.dim();
  bool flag_3d = false;
  const int bra_len = (sorb - 1) / 64 + 1;
  // notice: tensor_len： 是bra_tensor[1] 除去尾部0的长度
  const int tensor_len = (sorb - 1) / 8 + 1;
  if (ket_dim == 3) {
    flag_3d = true;
    // bra: (n, tensor_len), ket: (n, m, tensor_len)
    n = bra_tensor.size(0), m = ket_tensor.size(1);
  } else if (ket_dim == 2) {
    flag_3d = false;
    // bra: (n, tensor_len), ket: (m, tensor_len)
    n = bra_tensor.size(0), m = ket_tensor.size(0);
  } else {
    throw "bra dim error";
  }

  torch::Tensor Hmat = torch::zeros({n, m}, h1e_tensor.options());

  double *h1e_ptr = h1e_tensor.data_ptr<double>();
  double *h2e_ptr = h2e_tensor.data_ptr<double>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  unsigned long *ket_ptr =
      reinterpret_cast<unsigned long *>(ket_tensor.data_ptr<uint8_t>());
  double *Hmat_ptr = Hmat.data_ptr<double>();

  auto t2 = get_time();
  auto delta1 = get_duration_nano(t2 - t3);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "CPU Hmat initialization time: " << delta1 / 1000000 << " ms"
              << std::endl;
  }

  auto t0 = get_time();
  if (flag_3d) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // Hmat_ptr[i, j] = get_Hij_cpu(bra_ptr[i], ket[i, m])
        Hmat_ptr[i * m + j] = get_Hij_cpu(
            &bra_ptr[i * bra_len], &ket_ptr[i * m * bra_len + j * bra_len],
            h1e_ptr, h2e_ptr, sorb, nele, tensor_len, bra_len);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // Hmat_ptr[i, j] = get_Hij_cpu(bra_ptr[i], ket_ptr[m])
        Hmat_ptr[i * m + j] =
            get_Hij_cpu(&bra_ptr[i * bra_len], &ket_ptr[j * bra_len], h1e_ptr,
                        h2e_ptr, sorb, nele, tensor_len, bra_len);
      }
    }
  }

  auto t1 = get_time();
  auto delta = get_duration_nano(t1 - t0);
  if (VERBOSE) {
    std::cout << std::setprecision(6);
    std::cout << "CPU calculate <n|H|m> time: " << delta / 1000000 << " ms"
              << std::endl;
    std::cout << "Total CPU function time: "
              << get_duration_nano(t1 - t3) / 1000000 << " ms\n"
              << std::endl;
  }

  return Hmat;
}

torch::Tensor get_comb_tensor_cpu(torch::Tensor &bra_tensor, const int sorb,
                                  const int nele, bool ms_equal) {
  const int no = nele;
  const int nv = sorb - nele;
  const int bra_len = (sorb - 1) / 64 + 1;
  int ncomb = std::get<2>(get_nsingles_doubles(no, nv, ms_equal));
  const int nob = nele / 2, noa = no - nob;
  const int nvb = nv / 2, nva = nv - nvb;
  // TODO: how to achieve CPU to CUDA using torch::KCPU in *.cpp or *.cu file?
  const int batch = bra_tensor.size(0);
  const int dim = bra_tensor.dim();
  bool flag_3d = false;
  torch::Tensor comb;
  if ((dim == 1) or (batch == 1 && dim == 2)) {
    comb = torch::zeros({ncomb, 8 * bra_len}, bra_tensor.options());
  } else if (batch > 1 && dim == 2) {
    flag_3d = true;
    comb = torch::zeros({batch, ncomb, 8 * bra_len}, bra_tensor.options());
  } else {
    throw "bra dim may be error";
  }
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  if (flag_3d) {
    for (int i = 0; i < batch; i++) {
      // Notice the index in 3D tensor
      if (not ms_equal) {
        get_comb_2d(&bra_ptr[i], &comb_ptr[i * ncomb * bra_len], sorb, bra_len,
                    no, nv, false);
      } else {
        get_comb_2d(&bra_ptr[i], &comb_ptr[i * ncomb * bra_len], sorb, bra_len,
                    noa, nob, nva, nvb);
      }
    }
  } else {
    if (not ms_equal) {
      get_comb_2d(bra_ptr, comb_ptr, sorb, bra_len, no, nv, false);
    } else {
      get_comb_2d(bra_ptr, comb_ptr, sorb, bra_len, noa, nob, nva, nvb);
    }
  }
  return comb;
}

tuple_tensor_2d get_comb_tensor_cpu_1(torch::Tensor &bra_tensor, const int sorb,
                                      const int nele, const int noA,
                                      const int noB, bool flag_bit) {
  const int bra_len = (sorb - 1) / 64 + 1;
  const int ncomb = get_Num_SinglesDoubles(sorb, noA, noB) + 1;
  const int batch = bra_tensor.size(0);
  const int dim = bra_tensor.dim();
  bool flag_3d = false;
  torch::Tensor comb, comb_bit;

  if ((dim == 1) or (batch == 1 && dim == 2)) {
    // comb: [ncomb , 8 * bra_len]
    comb = bra_tensor.reshape({1, -1}).repeat({ncomb, 1});
    if (flag_bit) {
      auto x = bra_tensor.reshape({1, -1}); //lvalue
      comb_bit = uint8_to_bit_cpu(x, sorb).repeat({ncomb, 1});
    }
  } else if (batch > 1 && dim == 2) {
    flag_3d = true;
    // comb: [nSample, ncomb, 8 * bra_len]
    comb = bra_tensor.reshape({batch, 1, -1}).repeat({1, ncomb, 1});
    if (flag_bit) {
      //comb_bit [nSample, ncomb, sorb]
      comb_bit = uint8_to_bit_cpu(bra_tensor, sorb)
                     .reshape({batch, 1, -1})
                     .repeat({1, ncomb, 1});
    }
  } else {
    throw "bra dim may be error";
  }

  unsigned long *comb_ptr =
      reinterpret_cast<unsigned long *>(comb.data_ptr<uint8_t>());
  if (! flag_bit){
    comb_bit = torch::ones({1}, torch::TensorOptions().dtype(torch::kDouble));
  }
  double *comb_bit_ptr = comb_bit.data_ptr<double>();

  torch::Tensor merged = get_merged_olst_vlst(bra_tensor, nele, sorb, noA, noB);
  int *merged_ptr = merged.data_ptr<int32_t>();

  if (flag_3d) {
    for (int i = 0; i < batch; i++) {
      for (int j = 1; j < ncomb; j++) {
        if (flag_bit) {
          get_comb_2d(&comb_ptr[i * ncomb * bra_len + j * bra_len],
                      &comb_bit_ptr[i * ncomb * sorb + j * sorb], &merged_ptr[i * sorb],j-1, sorb,
                      bra_len, noA, noB);
        } else {
          get_comb_2d(&comb_ptr[i * ncomb * bra_len + j * bra_len], &merged_ptr[i*sorb],j-1, sorb,
                      bra_len, noA, noB);
        }
        // comb[i*ncomb * bra_len + j * bra_len], idx = j-1
        // comb_bit[i*ncomb * sorb + j * sorb]
      }
    }
  } else {
    for (int i = 1; i < ncomb; i++) {
      if (flag_bit) {
        get_comb_2d(&comb_ptr[i * bra_len], &comb_bit_ptr[i * sorb], merged_ptr,i - 1,
                    sorb, bra_len, noA, noB);
      } else {
        get_comb_2d(&comb_ptr[i * bra_len],merged_ptr, i - 1, sorb, bra_len, noA, noB);
      }
      // comb[i * bre_len], idx = i-1;
      // comb_bit[i * sorb]
    }
  }
  return std::make_tuple(comb, comb_bit);
}

// RBM
torch::Tensor uint8_to_bit_cpu(torch::Tensor &bra_tensor, const int sorb) {
  const int bra_len = (sorb - 1) / 64 + 1;
  const int bra_dim = bra_tensor.dim();
  int n = 0, m = 0;
  torch::Tensor comb_bit;
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat64)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);

  if (bra_dim == 3) {
    // [batch, ncomb, sorb]
    n = bra_tensor.size(0), m = bra_tensor.size(1);
    comb_bit = torch::zeros({n, m, sorb}, options);
  } else if (bra_dim == 2) {
    // [ncomb, sorb]
    n = bra_tensor.size(0);
    comb_bit = torch::zeros({n, sorb}, options);
  } else if (bra_dim == 1) {
    comb_bit = torch::zeros({sorb}, options);
  } else {
    throw "bra dim error";
  }

  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());
  double *comb_ptr = comb_bit.data_ptr<double>();

  if (bra_dim == 3) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        get_zvec_cpu(&bra_ptr[i * m * bra_len + j * bra_len],
                     &comb_ptr[i * m * sorb + j * sorb], sorb, bra_len);
      }
    }
  } else if (bra_dim == 2) {
    for (int i = 0; i < n; i++) {
      get_zvec_cpu(&bra_ptr[i * bra_len], &comb_ptr[i * sorb], sorb, bra_len);
    }
  } else if (bra_dim == 1) {
    get_zvec_cpu(bra_ptr, comb_ptr, sorb, bra_len);
  }

  return comb_bit;
}

std::tuple<torch::Tensor, torch::Tensor> get_olst_vlst_cpu(
    torch::Tensor &bra_tensor, const int sorb, const int nele) {
  const int no = nele;
  const int nv = sorb - nele;
  // const int tensor_len =(sorb-1)/8 + 1;
  const int bra_len = (sorb - 1) / 64 + 1;

  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(bra_tensor.layout())
                     .device(bra_tensor.device())
                     .requires_grad(false);
  torch::Tensor vlst = torch::zeros({nv}, options);
  torch::Tensor olst = torch::zeros({no}, options);
  int *vlst_ptr = vlst.data_ptr<int32_t>();
  int *olst_ptr = olst.data_ptr<int32_t>();
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());

  get_olst_cpu(bra_ptr, olst_ptr, bra_len);
  get_vlst_cpu(bra_ptr, vlst_ptr, sorb, bra_len);

  return std::make_tuple(olst, vlst);
}

// MCMC sampling in RBM
tuple_tensor_2d spin_flip_rand(
    torch::Tensor &bra_tensor, const int sorb, const int nele, const int seed) {
  const int no = nele;
  const int nv = sorb - nele;
  const int bra_len = (sorb - 1) / 64 + 1;

  // auto vlst = std::make_unique<int[]>(nv);
  int olst[MAX_NO] = {0};
  int vlst[MAX_NV] = {0};
  int olst_a[MAX_NOA] = {0};
  int olst_b[MAX_NOB] = {0};
  int vlst_a[MAX_NOA] = {0};
  int vlst_b[MAX_NOB] = {0};
  const int nob = nele / 2, noa = no - nob;
  const int nvb = nv / 2, nva = nv - nvb;
  unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());

  get_olst_cpu(bra_ptr, olst, olst_a, olst_b, bra_len);
  get_vlst_cpu(bra_ptr, vlst, vlst_a, vlst_b, sorb, bra_len);

  auto x = get_nsingles_doubles(no, nv, true);
  int nsingles = std::get<0>(x);
  int ndoubles = std::get<1>(x);
  int ncomb = nsingles + ndoubles;
  static std::mt19937 rng(seed);

  static std::uniform_int_distribution<int> u0(0, ncomb - 1);
  static std::uniform_int_distribution<int> us(0, no * nv - 1);
  int r = u0(rng);
  if (r < nsingles) {
    while (true) {
      int xs = us(rng);
      int i = olst[xs % no];
      int k = vlst[xs / no];
      // a->a/b->b
      if ((i & 1) == (k & 1)) {
        BIT_FLIP(bra_ptr[i / 64], i % 64);
        BIT_FLIP(bra_ptr[k / 64], k % 64);
        break;
      } else {
        continue;
      }
    }
  } else {
    if (DEBUG) {
      // This is pretty slow, only using test for MCMC sampling.
      while (true) {
        int xs = us(rng), xd = us(rng);
        int idi = xs % no, idj = xd % no;
        int idk = xs / no, idl = xd / no;
        int i = olst[idi], j = olst[idj];
        int k = vlst[idk], l = vlst[idl];
        // see get_comb_2d
        if (((i > j) && (k > l)) || ((i < j) && (k < l))) {
          ;
        } else {
          continue;
        }
        bool flag_one, flag_two;
        flag_one = (((i & 1) == (k & 1)) && ((j & 1) == (l & 1)));
        flag_two = (((i & 1) == (l & 1)) && ((j & 1) == (k & 1)));
        if (flag_one || flag_two) {
          BIT_FLIP(bra_ptr[i / 64], i % 64);
          BIT_FLIP(bra_ptr[j / 64], j % 64);
          BIT_FLIP(bra_ptr[k / 64], k % 64);
          BIT_FLIP(bra_ptr[l / 64], l % 64);
          break;
        } else {
          continue;
        }
      }
    } else {
      int Naa = noa * (noa - 1) * nva * (nva - 1) / 4;
      int Nbb = nob * (nob - 1) * nvb * (nvb - 1) / 4;
      int Nab = noa * nva * nob * nob;
      int i = 0, j = 0, k = 0, l = 0;
      bool flag = false;
      static std::uniform_int_distribution<int> ud(0, Naa + Nbb + Nab - 1);
      int m = ud(rng);
      if (m < Naa) {
        static std::uniform_int_distribution<int> uaa(0, noa * nva - 1);
        while (not flag) {
          int xs = uaa(rng), xd = uaa(rng);
          int idi = xs % noa, idj = xd % nva;
          int idk = xs / noa, idl = xd / nva;
          i = olst_a[idi], j = olst_a[idj];
          k = vlst_a[idk], l = vlst_a[idl];
          if ((i != j) && (k != l)) {
            flag = true;
          }
        }
      } else if (m >= Naa and m < Naa + Nbb) {
        static std::uniform_int_distribution<int> ubb(0, nob * nvb - 1);
        while (not flag) {
          int xs = ubb(rng), xd = ubb(rng);
          int idi = xs % nob, idj = xd % nvb;
          int idk = xs / nob, idl = xd / nvb;
          i = olst_b[idi], j = olst_b[idj];
          k = vlst_b[idk], l = vlst_b[idl];
          if ((i != j) && (k != l)) {
            flag = true;
          }
        }
      } else {
        int m1 = m - Naa - Nbb;
        int idij = m1 % (noa * nob);
        int idkl = m1 / (noa * nob);
        i = olst_a[idij % noa];
        j = olst_b[idij / noa];
        k = vlst_a[idkl % nva];
        l = vlst_b[idkl / nva];
      }
      BIT_FLIP(bra_ptr[i / 64], i % 64);
      BIT_FLIP(bra_ptr[j / 64], j % 64);
      BIT_FLIP(bra_ptr[k / 64], k % 64);
      BIT_FLIP(bra_ptr[l / 64], l % 64);
    }
  }
  return std::make_tuple(uint8_to_bit_cpu(bra_tensor, sorb), bra_tensor);
}


tuple_tensor_2d spin_flip_rand_1(
    torch::Tensor &bra_tensor, const int sorb, const int nele, const int noA,
    const int noB, const int seed){
  const int bra_len = (sorb - 1) / 64 + 1;
  int merged[MAX_NO + MAX_NV] = {0};
  unsigned long *bra_ptr =
    reinterpret_cast<unsigned long *>(bra_tensor.data_ptr<uint8_t>());

  get_olst_cpu_ab(bra_ptr, merged, bra_len);
  get_vlst_cpu_ab(bra_ptr, merged +nele, sorb, bra_len);
  const int ncomb = get_Num_SinglesDoubles(sorb, noA, noB);
  static std::mt19937 rng(seed);
  static std::uniform_int_distribution<int> u0(0, ncomb - 1);
  int r0 = u0(rng);
  int idx_lst[4] = {0};
  unpack_Singles_Doubles(sorb, noA, noB, r0, idx_lst);
  for (int i = 0; i < 4; i++){
    int idx = merged[idx_lst[i]]; //merged[olst, vlst]
    BIT_FLIP(bra_ptr[idx/64], idx % 64);
  }
  return std::make_tuple(uint8_to_bit_cpu(bra_tensor, sorb), bra_tensor);
}