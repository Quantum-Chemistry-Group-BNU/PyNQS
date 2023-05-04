#include <cstddef>

#include "excitation_cuda.h"
#include "hamiltonian_cuda.h"
#include "kernel.h"
#include "onstate_cuda.h"

#include "../common/default.h"

__global__ void pack_states_kernel(const uint8_t *bra, uint8_t *states,
                                   const int sorb, const int bra_len,
                                   const int m, const size_t n) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= n)
    return;
  int idx = idn % m, idy = idn / m;
  for (int i = 0; i < 8 && idx * 8 + i < sorb; i++) {
    uint8_t value = reinterpret_cast<uint8_t>(bra[sorb * idy + idx * 8 + i]);
    states[idy * 8 * bra_len + idx] |= (value << i);
  }
}

__host__ void squant::pack_states_cuda(uint8_t *states, const uint8_t *bra,
                                       const int sorb, const int nbatch,
                                       const int bra_len,
                                       const int tensor_len) {

  dim3 threads(1024);
  dim3 blocks((tensor_len * nbatch + threads.x - 1) / threads.x);
  pack_states_kernel<<<blocks, threads>>>(bra, states, sorb, bra_len,
                                          tensor_len, tensor_len * nbatch);
}

__host__ __global__ void get_zvec_kernel(double *comb, const unsigned long *bra,
                                         const int sorb, const int bra_len,
                                         const size_t m) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= m)
    return;
  size_t idm = idn / sorb;
  // TODO: the cost of launching thread.
  squant::get_zvec_cuda(&bra[idm * bra_len], &comb[idm * sorb], sorb, bra_len,
                        idn % sorb);
}

__host__ void squant::unpack_states_cuda(double *comb, const unsigned long *bra,
                                         const int sorb, const int bra_len,
                                         const int nbatch, const size_t numel) {
  dim3 threads(1024);
  dim3 blocks((numel + threads.x - 1) / threads.x);
  get_zvec_kernel<<<blocks, threads>>>(comb, bra, sorb, bra_len, numel);
}

// <i|H|j> matrix, i,j: 2D (nbatch, onv)
// construct Hij matrix -> (nbatch, nbatch)
__global__ void get_Hij_kernel_2D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t tensor_len,
                                  const size_t bra_len, const int n,
                                  const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] =
      squant::get_Hij_cuda(&bra[idn * bra_len], &ket[idm * bra_len], h1e, h2e,
                           sorb, nele, tensor_len, bra_len);
}

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv)
// local energy -> (nbatch, ncomb)
__global__ void get_Hij_kernel_3D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t tensor_len,
                                  const size_t bra_len, const int n,
                                  const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idn * m * bra_len + idm * bra_len], h1e, h2e,
      sorb, nele, tensor_len, bra_len);
}

__host__ void squant::get_Hij_3D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int tensor_len, const int bra_len,
                                      const int nbatch, const int ncomb) {
  dim3 threads(THREAD, THREAD);
  dim3 blocks((nbatch + threads.x - 1) / threads.x,
              (ncomb + threads.y - 1) / threads.y);
  get_Hij_kernel_3D<<<blocks, threads>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                         tensor_len, bra_len, nbatch, ncomb);
}

__host__ void squant::get_Hij_2D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int tensor_len, const int bra_len,
                                      const int n, const int m) {
  dim3 threads(THREAD, THREAD);
  dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  get_Hij_kernel_2D<<<blocks, threads>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                         tensor_len, bra_len, n, m)
}

__global__ void get_merged_ovlst_kernel(const unsigned long *bra, int *merged,
                                        int sorb, int nele, int bra_len,
                                        int n) {
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= n)
    return;
  squant::get_ovlst_cuda(&bra[idm * bra_len], &merged[idm * sorb], sorb, nele,
                         bra_len);
}

__host__ void squant::get_merged_cuda(const unsigned long *bra, int *merged,
                                      const int sorb, const int nele,
                                      const int bra_len, const int nbatch) {
  dim3 threads(1024);
  dim3 blocks((nbatch + threads.x - 1) / threads.x);
  get_merged_ovlst_kernel<<<blocks, threads>>>(bra, merged, sorb, nele, bra_len,
                                               nbatch);
}

__global__ void get_comb_SD_kernel(unsigned long *comb, double *comb_bit,
                                   const int *merged, int sorb, int bra_len,
                                   int noA, int noB, int nbatch, int ncomb) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbatch || idm >= ncomb || idm == 0)
    return;
  squant::get_comb_SD_cuda(&comb[idn * ncomb * bra_len + idm * bra_len],
                           &comb_bit[idn * ncomb * sorb + idm * sorb],
                           &merged[idn * sorb], idm - 1, sorb, noA, noB);
}

__global__ void get_comb_SD_kernel(unsigned long *comb, const int *merged,
                                   const int sorb, const int bra_len,
                                   const int noA, const int noB,
                                   const int nbatch, const int ncomb) {
  const int idn = blockIdx.x * blockDim.x + threadIdx.x;
  const int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbatch || idm >= ncomb || idm == 0)
    return;
  squant::get_comb_SD_cuda(&comb[idn * ncomb * bra_len + idm * bra_len],
                           &merged[idn * sorb], idm - 1, sorb, noA, noB);
}

__host__ void squant::get_comb_cuda(unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  dim3 threads(THREAD, THREAD);
  dim3 blocks((nbatch + threads.x - 1) / threads.x,
              (ncomb + threads.y - 1) / threads.y);
  get_comb_SD_kernel<<<blocks, threads>>>(comb, merged_ovlst, sorb, bra_len,
                                          noA, noB, nbatch, ncomb);
}

__host__ void squant::get_comb_cuda(double *comb_bit, unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  dim3 threads(THREAD, THREAD);
  dim3 blocks((nbatch + threads.x - 1) / threads.x,
              (ncomb + threads.y - 1) / threads.y);
  get_comb_SD_kernel<<<blocks, threads>>>(comb, comb_bit, merged_ovlst, sorb,
                                          bra_len, noA, noB, nbatch, ncomb);
}