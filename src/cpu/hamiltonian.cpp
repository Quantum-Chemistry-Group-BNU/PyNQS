#include "hamiltonian.h"

#include "onstate.h"

namespace squant {

template<typename T>
T h1e_get_cpu(const T *h1e, const size_t i, const size_t j,
                   const size_t sorb) {
  return h1e[j * sorb + i];
}

template<typename T>
T h2e_get_cpu(const T *h2e, const size_t i, const size_t j,
                   const size_t k, const size_t l) {
  if ((i == j) || (k == l)) return 0.00;
  size_t ij = i > j ? i * (i - 1) / 2 + j : j * (j - 1) / 2 + i;
  size_t kl = k > l ? k * (k - 1) / 2 + l : l * (l - 1) / 2 + k;
  T sgn = 1;
  sgn = i > j ? sgn : -sgn;
  sgn = k > l ? sgn : -sgn;
  T val;
  if (ij >= kl) {
    size_t ijkl = ij * (ij + 1) / 2 + kl;
    val = sgn * h2e[ijkl];
  } else {
    size_t ijkl = kl * (kl + 1) / 2 + ij;
    val = sgn * h2e[ijkl];  // sgn * conjugate(h2e[ijkl])
  }
  return val;
}

template<typename T>
T get_Hii_cpu(const unsigned long *bra, const unsigned long *ket,
                   const T *h1e, const T *h2e, const int sorb,
                   const int nele, const int bra_len) {
  T Hii = 0.00;
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
  return Hii;
}

template<typename T>
T get_HijS_cpu(const unsigned long *bra, const unsigned long *ket,
                    const T *h1e, const T *h2e, const size_t sorb,
                    const int bra_len) {
  T Hij = 0.00;
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
  Hij *= static_cast<T>(sgn);
  return Hij;
}

template<typename T>
T get_HijD_cpu(const unsigned long *bra, const unsigned long *ket,
                    const T *h1e, const T *h2e, const size_t sorb,
                    const int bra_len) {
  int p[2], q[2];
  diff_orb_cpu(bra, ket, bra_len, p, q);
  int sgn = parity_cpu(bra, p[0]) * parity_cpu(bra, p[1]) *
            parity_cpu(ket, q[0]) * parity_cpu(ket, q[1]);
  T Hij = h2e_get_cpu(h2e, p[0], p[1], q[0], q[1]);
  Hij *= static_cast<T>(sgn);
  return Hij;
}

template<typename T>
T get_Hij_cpu(const unsigned long *bra, const unsigned long *ket,
                   const T *h1e, const T *h2e, const size_t sorb,
                   const int nele, const int bra_len) {
  T Hij = 0.00;
  int type[2] = {0};
  diff_type_cpu(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    Hij = get_Hii_cpu(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    Hij = get_HijS_cpu(bra, ket, h1e, h2e, sorb, bra_len);
  } else if (type[0] == 2 && type[1] == 2) {
    Hij = get_HijD_cpu(bra, ket, h1e, h2e, sorb, bra_len);
  }
  return Hij;
}

template float h1e_get_cpu<float>(const float *, size_t, size_t, size_t);
template double h1e_get_cpu<double>(const double *, size_t, size_t, size_t);

template float h2e_get_cpu<float>(const float *, size_t, size_t, size_t,
                                  size_t);
template double h2e_get_cpu<double>(const double *, size_t, size_t, size_t,
                                    size_t);

template float get_Hii_cpu<float>(const unsigned long *, const unsigned long *,
                                  const float *, const float *, int, int, int);
template double get_Hii_cpu<double>(const unsigned long *,
                                    const unsigned long *, const double *,
                                    const double *, int, int, int);

template float get_HijS_cpu<float>(const unsigned long *, const unsigned long *,
                                   const float *, const float *, size_t, int);
template double get_HijS_cpu<double>(const unsigned long *,
                                     const unsigned long *, const double *,
                                     const double *, size_t, int);

template float get_HijD_cpu<float>(const unsigned long *, const unsigned long *,
                                   const float *, const float *, size_t, int);
template double get_HijD_cpu<double>(const unsigned long *,
                                     const unsigned long *, const double *,
                                     const double *, size_t, int);

template float get_Hij_cpu<float>(const unsigned long *, const unsigned long *,
                                  const float *, const float *, size_t, int,
                                  int);
template double get_Hij_cpu<double>(const unsigned long *,
                                    const unsigned long *, const double *,
                                    const double *, size_t, int, int);

}  // namespace squant