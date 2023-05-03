#include "../common/utils_cuda.h"

namespace squant{

__device__ double h1e_get_cuda(double *h1e, size_t i, size_t j, size_t sorb);

__device__ double h2e_get_cuda(double *h2e, size_t i, size_t j, size_t k, size_t l);

__device__ double get_Hii_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, int sorb, const int nele, int bra_len);

__device__ double get_HijS_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

__device__ double get_HijD_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                    double *h2e, size_t sorb, int bra_len);

__device__ double get_Hij_cuda(unsigned long *bra, unsigned long *ket, double *h1e,
                   double *h2e, size_t sorb, size_t nele, size_t tensor_len,
                   size_t bra_len);

} // namespace squant