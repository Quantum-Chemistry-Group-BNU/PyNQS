#pragma once
#include "../common/utils.h"

namespace squant{

double h1e_get_cpu(double *h1e, size_t i, size_t j, size_t sorb);

double h2e_get_cpu(double *h2e, size_t i, size_t j, size_t k, size_t l);

double get_Hii_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, int sorb, const int nele, int bra_len);

double get_HijS_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

double get_HijD_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

double get_Hij_cpu(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, size_t sorb, size_t nele, size_t tensor_len,
                   size_t bra_len);

} // namespace squant