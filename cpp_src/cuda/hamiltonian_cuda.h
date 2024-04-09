#pragma once
#include <cstddef>

#include "utils_cuda.h"

namespace squant {

__device__ double h1e_get_cuda(const double *h1e, const size_t i,
                               const size_t j, const size_t sorb);

__device__ double h2e_get_cuda(const double *h2e, const size_t i,
                               const size_t j, const size_t k, const size_t l);

__device__ double get_Hii_cuda(const unsigned long *bra,
                               const unsigned long *ket, const double *h1e,
                               const double *h2e, const size_t sorb,
                               const int nele, const int bra_len);

__device__ double get_HijS_cuda(const unsigned long *bra,
                                const unsigned long *ket, const double *h1e,
                                const double *h2e, const size_t sorb,
                                const int bra_len);

__device__ double get_HijD_cuda(const unsigned long *bra,
                                const unsigned long *ket, const double *h1e,
                                const double *h2e, const size_t sorb,
                                const int bra_len);

__device__ double get_Hij_cuda(const unsigned long *bra,
                               const unsigned long *ket, const double *h1e,
                               const double *h2e, const size_t sorb,
                               const int nele, const int bra_len);

} // namespace squant