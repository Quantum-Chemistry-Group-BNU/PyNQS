#pragma once
#include "../common/utils.h"

namespace squant {

template<typename T>
T h1e_get_cpu(const T *h1e, const size_t i, const size_t j,
                   const size_t sorb);

template<typename T>
T h2e_get_cpu(const T *h2e, const size_t i, const size_t j,
                   const size_t k, const size_t l);

template<typename T>
T get_Hii_cpu(const unsigned long *bra, const unsigned long *ket,
                   const T *h1e, const T *h2e, const int sorb,
                   const int nele, const int bra_len);

template<typename T>
T get_HijS_cpu(const unsigned long *bra, const unsigned long *ket,
                    const T *h1e, const T *h2e, const size_t sorb,
                    const int bra_len);

template<typename T>
T get_HijD_cpu(const unsigned long *bra, const unsigned long *ket,
                    const T *h1e, const T *h2e, const size_t sorb,
                    const int bra_len);

template<typename T>
T get_Hij_cpu(const unsigned long *bra, const unsigned long *ket,
                   const T *h1e, const T *h2e, const size_t sorb,
                   const int nele, const int bra_len);

}  // namespace squant