#include "hamiltonian.h"
#include "onstate.h"

NAMESPACE_BEGIN(fock)

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
  int sgn =parity_cpu(bra, p[0]) * parity_cpu(ket, q[0]);
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

NAMESPACE_END(fock)