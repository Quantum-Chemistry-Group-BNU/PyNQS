#include "spin_flip.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "cuda_handle_error.h" // gcc 13 error, compile using gcc 11
#include "excitation_cuda.h"
#include "hamiltonian_cuda.h"
#include "kernel.h"
#include "onstate_cuda.h"
#include <curand_kernel.h>

#include "../common/default.h"

namespace {
constexpr unsigned long kEvenBitMask = 0x5555555555555555ULL;
constexpr unsigned long kOddBitMask = 0xAAAAAAAAAAAAAAAAULL;

template <int _len>
__device__ inline int select_spin_rank_orbital(const unsigned long *bra,
                                               const int sorb, const int spin,
                                               const bool occupied, int rank) {
  const int valid_bits_last = sorb - ((_len - 1) << 6);
  const unsigned long tail_mask =
      squant::get_ones_cuda(valid_bits_last == 64 ? 64 : valid_bits_last);
  const unsigned long spin_mask = spin ? kOddBitMask : kEvenBitMask;

#pragma unroll
  for (int word = 0; word < _len; ++word) {
    unsigned long bits = occupied ? bra[word] : ~bra[word];
    bits &= (word == _len - 1) ? tail_mask : ~0ULL;
    bits &= spin_mask;

    int cnt = __popcll(bits);
    if (rank >= cnt) {
      rank -= cnt;
      continue;
    }

    while (rank > 0) {
      bits &= bits - 1ULL;
      --rank;
    }

    int bit = __ffsll(static_cast<long long>(bits)) - 1;
    return (word << 6) + bit;
  }

  return 0;
}

template <int _len>
__device__ inline int resolve_merged_rank(const unsigned long *bra,
                                          const int sorb, const int noA,
                                          const int noB,
                                          const int merged_rank) {
  const int spin = merged_rank & 1;
  const int rank = merged_rank >> 1;
  const int nocc = spin ? noB : noA;
  const bool occupied = rank < nocc;
  const int target_rank = occupied ? rank : (rank - nocc);
  return select_spin_rank_orbital<_len>(bra, sorb, spin, occupied,
                                        target_rank);
}
}  // namespace

template <int _len>
__global__ void
spin_flip_rand_kernel_philox(unsigned long *bra, const int16_t *merged,
                             const int sorb, const int nele, const int noA,
                             const int noB, const int64_t nbatch,
                             const int ncomb, at::PhiloxCudaState args) {
  (void)nele;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nbatch)
    return;

  auto [seed, offset] = at::cuda::philox::unpack(args);
  curandStatePhilox4_32_10_t state;
  curand_init((unsigned long)seed, (unsigned long)idx, (unsigned long)offset,
              &state);

  unsigned int m = (unsigned int)(ncomb + 1);
  unsigned int threshold = (unsigned int)((-(int64_t)m) % m);
  unsigned int r;
  do {
    r = curand(&state);
  } while (r < threshold);
  unsigned int r0 = r % m; // r0 in [0, ncomb]

  if (r0 == 0)
    return;

  int idx_lst[5] = {0};
  auto offset0 = idx * (int64_t)sorb;
  auto offset1 = idx * (int64_t)_len;

  squant::unpack_SinglesDoubles_cuda(sorb, noA, noB, (int)r0 - 1, idx_lst);

  for (int j = 0; j < 4; j++) {
    auto idy = merged[offset0 + idx_lst[j]];
    BIT_FLIP(bra[offset1 + idy / 64], idy % 64);
  }
}

template <int _len>
__global__ void spin_flip_rand_kernel_philox_direct(
    unsigned long *bra, const int sorb, const int noA, const int noB,
    const int64_t nbatch, const int ncomb, at::PhiloxCudaState args) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nbatch) {
    return;
  }

  auto [seed, offset] = at::cuda::philox::unpack(args);
  curandStatePhilox4_32_10_t state;
  curand_init((unsigned long)seed, (unsigned long)idx, (unsigned long)offset,
              &state);

  const unsigned int m = static_cast<unsigned int>(ncomb + 1);
  const unsigned int threshold = static_cast<unsigned int>((-(int64_t)m) % m);
  unsigned int r = 0;
  do {
    r = curand(&state);
  } while (r < threshold);
  const unsigned int r0 = r % m;  // r0 in [0, ncomb]
  if (r0 == 0) {
    return;
  }

  int idx_lst[5] = {0};
  squant::unpack_SinglesDoubles_cuda(sorb, noA, noB, (int)r0 - 1, idx_lst);

  unsigned long *bra_ptr = &bra[idx * (int64_t)_len];
  int orbital_idx[4] = {0, 0, 0, 0};
  int last_rank = -1;
  int last_orbital = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int rank = idx_lst[j];
    int orbital = last_orbital;
    if (rank != last_rank) {
      orbital = resolve_merged_rank<_len>(bra_ptr, sorb, noA, noB, rank);
      last_rank = rank;
      last_orbital = orbital;
    }
    orbital_idx[j] = orbital;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int orbital = orbital_idx[j];
    BIT_FLIP(bra_ptr[orbital >> 6], orbital & 63);
  }
}

__host__ void spin_flip_rand_philox_impl(unsigned long *bra,
                                         const int16_t *merged, const int sorb,
                                         const int nele, const int noA,
                                         const int noB, const int64_t nbatch,
                                         const int ncomb,
                                         at::PhiloxCudaState philox_args) {
  dim3 blockDim(256);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  spin_flip_rand_kernel_philox<MAX_SORB_LEN><<<gridDim, blockDim>>>(
      bra, merged, sorb, nele, noA, noB, nbatch, ncomb, philox_args);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__host__ void spin_flip_rand_philox_direct_impl(
    unsigned long *bra, const int sorb, const int noA, const int noB,
    const int64_t nbatch, const int ncomb, at::PhiloxCudaState philox_args) {
  dim3 blockDim(256);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  spin_flip_rand_kernel_philox_direct<MAX_SORB_LEN>
      <<<gridDim, blockDim>>>(bra, sorb, noA, noB, nbatch, ncomb, philox_args);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}
