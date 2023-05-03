#include "../common/default.h"
#include "hamiltonian_cuda.h"
#include "onstate_cuda.h"

namespace squant {

__device__ double h1e_get_cuda(double *h1e, size_t i, size_t j, size_t sorb) {
  return h1e[j * sorb + i];
}

__device__ double h2e_get_cuda(double *h2e, size_t i, size_t j, size_t k, size_t l) {
  if ((i == j) || (k == l))
    return 0.00;
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
    val = sgn * h2e[ijkl]; // sgn * conjugate(h2e[ijkl])
  }
  return val;
}

__device__ double get_Hii_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                          double *h2e, int sorb, const int nele, int bra_len) {
  double Hii = 0.00;
  int olst[MAX_NELE] = {0};
  get_olst_cuda(bra, olst, bra_len);

  for (int i = 0; i < nele; i++) {
    int p = olst[i]; //<p|h|p>
    Hii += h1e_get_cuda(h1e, p, p, sorb);
    for (int j = 0; j < i; j++) {
      int q = olst[j];
      Hii += h2e_get_cuda(h2e, p, q, p, q); //<pq||pq> Storage not continuous
    }
  }
  return Hii;
}

__device__ double get_HijS_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                           double *h2e, size_t sorb, int bra_len) {
  double Hij = 0.00;
  int p[1], q[1];
  diff_orb_cuda(bra, ket, bra_len, p, q);
  Hij += h1e_get_cuda(h1e, p[0], q[0], sorb); // hpq
  for (int i = 0; i < bra_len; i++) {
    unsigned long repr = bra[i];
    while (repr != 0) {
      int j = 63 - __clzl(repr);
      int k = 64 * i + j;
      Hij += h2e_get_cuda(h2e, p[0], k, q[0], k); //<pk||qk>
      repr &= ~(1ULL << j);
    }
  }
  int sgn = parity_cuda(bra, p[0]) * parity_cuda(ket, q[0]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

__device__ double get_HijD_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                           double *h2e, size_t sorb, int bra_len) {
  int p[2], q[2];
  diff_orb_cuda(bra, ket, bra_len, p, q);
  int sgn = parity_cuda(bra, p[0]) * parity_cuda(bra, p[1]) * parity_cuda(ket, q[0]) *
            parity_cuda(ket, q[1]);
  double Hij = h2e_get_cuda(h2e, p[0], p[1], q[0], q[1]);
  Hij *= static_cast<double>(sgn);
  return Hij;
}

__device__ double get_Hij_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                          double *h2e, size_t sorb, size_t nele,
                          size_t tensor_len, size_t bra_len) {
  double Hij = 0.00;

  int type[2] = {0};
  diff_type_cuda(bra, ket, type, bra_len);
  if (type[0] == 0 && type[1] == 0) {
    Hij = get_Hii_cuda(bra, ket, h1e, h2e, sorb, nele, bra_len);
  } else if (type[0] == 1 && type[1] == 1) {
    Hij = get_HijS_cuda(bra, ket, h1e, h2e, sorb, bra_len);
  } else if (type[0] == 2 && type[1] == 2) {
    Hij = get_HijD_cuda(bra, ket, h1e, h2e, sorb, bra_len);
  }
  return Hij;
}

} // namespace squant