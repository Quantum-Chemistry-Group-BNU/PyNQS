#pragma once
#include "../common/utils.h"

namespace squant {

double h1e_get_cpu(const double *h1e, const size_t i, const size_t j,
                   const size_t sorb);

double h2e_get_cpu(const double *h2e, const size_t i, const size_t j,
                   const size_t k, const size_t l);

double get_Hii_cpu(const unsigned long *bra, const unsigned long *ket,
                   const double *h1e, const double *h2e, const int sorb,
                   const int nele, const int bra_len);

double get_HijS_cpu(const unsigned long *bra, const unsigned long *ket,
                    const double *h1e, const double *h2e, const size_t sorb,
                    const int bra_len);

double get_HijD_cpu(const unsigned long *bra, const unsigned long *ket,
                    const double *h1e, const double *h2e, const size_t sorb,
                    const int bra_len);

double get_Hij_cpu(const unsigned long *bra, const unsigned long *ket,
                   const double *h1e, const double *h2e, const size_t sorb,
                   const int nele, const int bra_len);

}  // namespace squant