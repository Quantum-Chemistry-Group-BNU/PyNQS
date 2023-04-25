#pragma once
#include "utils.h"

NAMESPACE_BEGIN(fock)

inline __device__ __host__ double h1e_get_cpu(double *h1e, size_t i, size_t j, size_t sorb) {
  return h1e[j * sorb + i];
}

inline __device__ __host__ double h2e_get_cpu(double *h2e, size_t i, size_t j, size_t k, size_t l) {
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

NAMESPACE_END(fock)