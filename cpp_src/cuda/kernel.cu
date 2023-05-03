#include "../common/utils_cuda.h"
#include "excitation_cuda.h"
#include "hamiltonian_cuda.h"
#include "onstate_cuda.h"

__global__ void pack_states_kernel(uint8_t *bra_ptr, uint8_t *states_ptr,
                                   const int sorb, const int bra_len,
                                   const int m, const size_t n) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= n)
    return;
  int idx = idn % m, idy = idn / m;
  for (int i = 0; i < 8 && idx * 8 + i < sorb; i++) {
    uint8_t value =
        reinterpret_cast<uint8_t>(bra_ptr[sorb * idy + idx * 8 + i]);
    states_ptr[idy * 8 * bra_len + idx] |= (value << i);
  }
}

__global__ void get_zvec_kernel_new(double *comb_ptr, unsigned long *bra,
                                    const int sorb, const int bra_len,
                                    size_t m) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= m)
    return;
  size_t idm = idn / sorb;
  squant::get_zvec_cuda(&bra[idm * bra_len], &comb_ptr[idm * sorb], sorb,
                        bra_len, idn % sorb);
}

// <i|H|j> matrix, i,j: 2D (nbatch, onv)
// construct Hij matrix -> (nbatch, nbatch)
__global__ void get_Hij_kernel_2D(double *Hmat_ptr, unsigned long *bra,
                                  unsigned long *ket, double *h1e, double *h2e,
                                  const size_t sorb, const size_t nele,
                                  const size_t tensor_len, const size_t bra_len,
                                  int n, int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;

  Hmat_ptr[idn * m + idm] =
      squant::get_Hij_cuda(&bra[idn * bra_len], &ket[idm * bra_len], h1e, h2e,
                           sorb, nele, tensor_len, bra_len);
}

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv) 
// local energy -> (nbatch, ncomb)
__global__ void get_Hij_kernel_3D(double *Hmat_ptr, unsigned long *bra,
                                  unsigned long *ket, double *h1e, double *h2e,
                                  const size_t sorb, const size_t nele,
                                  const size_t tensor_len, const size_t bra_len,
                                  int n, int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat_ptr[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idn * m * bra_len + idm * bra_len], h1e, h2e,
      sorb, nele, tensor_len, bra_len);
}

__global__ void get_comb_SD_kernel(unsigned long *comb_ptr, double *comb_bit_ptr,
                                int *merged_ptr, int sorb, int bra_len, int noA,
                                int noB, int nbtach, int ncomb) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbtach || idm >= ncomb || idm == 0)
    return;
  squant::get_comb_SD_cuda(&comb_ptr[idn * ncomb * bra_len + idm * bra_len],
                           &comb_bit_ptr[idn * ncomb * sorb + idm * sorb],
                           &merged_ptr[idn * sorb], idm - 1, sorb, bra_len, noA,
                           noB);
}

__global__ void get_comb_SD_kernel(unsigned long *comb_ptr, int *merged_ptr,
                                int sorb, int bra_len, int noA, int noB,
                                int nbtach, int ncomb) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbtach || idm >= ncomb || idm == 0)
    return;
  squant::get_comb_SD_cuda(&comb_ptr[idn * ncomb * bra_len + idm * bra_len],
                           &merged_ptr[idn * sorb], idm - 1, sorb, bra_len, noA,
                           noB);
}