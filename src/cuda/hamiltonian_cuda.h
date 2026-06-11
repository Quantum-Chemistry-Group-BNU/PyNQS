#pragma once
#include <cstddef>

#include "utils_cuda.h"

namespace squant {

template <typename T>
__device__ T h1e_get_cuda(const T *h1e, const size_t i,
                               const size_t j, const size_t sorb);

template <typename T>
__device__ T h2e_get_cuda(const T *h2e, const size_t i,
                               const size_t j, const size_t k, const size_t l);

template <typename T>
__device__ T get_Hii_cuda(const unsigned long *bra,
                               const unsigned long *ket, const T *h1e,
                               const T *h2e, const size_t sorb,
                               const int nele, const int bra_len);

template <typename T>
__device__ T get_HijS_cuda(const unsigned long *bra,
                                const unsigned long *ket, const T *h1e,
                                const T *h2e, const size_t sorb,
                                const int bra_len);

template <typename T>
__device__ T get_HijD_cuda(const unsigned long *bra,
                                const unsigned long *ket, const T *h1e,
                                const T *h2e, const size_t sorb,
                                const int bra_len);

template <typename T>
__device__ T get_Hij_cuda(const unsigned long *bra,
                               const unsigned long *ket, const T *h1e,
                               const T *h2e, const size_t sorb,
                               const int nele, const int bra_len);

} // namespace squant