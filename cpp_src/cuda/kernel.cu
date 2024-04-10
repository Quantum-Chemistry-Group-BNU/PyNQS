#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "cuda_handle_error.h" // gcc 13 error, compile using gcc 11
#include "excitation_cuda.h"
#include "hamiltonian_cuda.h"
#include "kernel.h"
#include "onstate_cuda.h"

#include "../common/default.h"

__global__ void tensor_to_onv_kernel(const uint8_t *bra, uint8_t *states,
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

// tensor(uint8): [1, 1, 0, 0] -> onv(uint8): 0b0011, 1 occupied, 0 unoccupied
__host__ void squant::tensor_to_onv_cuda(uint8_t *states, const uint8_t *bra,
                                         const int sorb, const int nbatch,
                                         const int bra_len,
                                         const int tensor_len) {
  dim3 blockDim(1024);
  dim3 gridDim((tensor_len * nbatch + blockDim.x - 1) / blockDim.x);
  tensor_to_onv_kernel<<<gridDim, blockDim>>>(bra, states, sorb, bra_len,
                                              tensor_len, tensor_len * nbatch);
}

__global__ void onv_to_tensor_kernel(double *comb, const unsigned long *bra,
                                     const int sorb, const int bra_len,
                                     const size_t m) {
  size_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= m)
    return;
  size_t idm = idn / sorb;
  // FIXME: the cost of launching thread. thread > 2**31
  squant::get_zvec_cuda(&bra[idm * bra_len], &comb[idm * sorb], sorb, bra_len,
                        idn % sorb);
}

// onv(unsinged long): 0b0011 -> tensor(double): [1.0. 1.0, 0.0, 0.0],
// 1: occupied, 0: unoccupied
__host__ void squant::onv_to_tensor_cuda(double *comb, const unsigned long *bra,
                                         const int sorb, const int bra_len,
                                         const int nbatch, const size_t numel) {
  dim3 blockDim(1024);
  dim3 gridDim((numel + blockDim.x - 1) / blockDim.x);
  // FIXME: numel > 2*
  onv_to_tensor_kernel<<<gridDim, blockDim>>>(comb, bra, sorb, bra_len, numel);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_Hij_kernel_2D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t bra_len,
                                  const int n, const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idm * bra_len], h1e, h2e, sorb, nele, bra_len);
}

__global__ void get_Hij_kernel_3D(double *Hmat, const unsigned long *bra,
                                  const unsigned long *ket, const double *h1e,
                                  const double *h2e, const size_t sorb,
                                  const size_t nele, const size_t bra_len,
                                  const int n, const int m) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= n || idm >= m)
    return;
  Hmat[idn * m + idm] = squant::get_Hij_cuda(
      &bra[idn * bra_len], &ket[idn * m * bra_len + idm * bra_len], h1e, h2e,
      sorb, nele, bra_len);
}

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv)
// local energy -> (nbatch, ncomb)
__host__ void squant::get_Hij_3D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int bra_len, const int nbatch,
                                      const int ncomb) {
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_Hij_kernel_3D<<<gridDim, blockDim>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                           bra_len, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

// <i|H|j> matrix, i,j: 2D (nbatch, onv)
// construct Hij matrix -> (nbatch1, nbatch2)
__host__ void squant::get_Hij_2D_cuda(double *Hmat, const unsigned long *bra,
                                      const unsigned long *ket,
                                      const double *h1e, const double *h2e,
                                      const int sorb, const int nele,
                                      const int bra_len, const int n,
                                      const int m) {
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (m + blockDim.y - 1) / blockDim.y);
  get_Hij_kernel_2D<<<gridDim, blockDim>>>(Hmat, bra, ket, h1e, h2e, sorb, nele,
                                           bra_len, n, m);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
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
  dim3 blockDim(1024);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  get_merged_ovlst_kernel<<<gridDim, blockDim>>>(bra, merged, sorb, nele,
                                                 bra_len, nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_comb_SD_kernel(unsigned long *comb, double *comb_bit,
                                   const int *merged, int sorb, int bra_len,
                                   int noA, int noB, int nbatch, int ncomb) {
  int idn = blockIdx.x * blockDim.x + threadIdx.x;
  int idm = blockIdx.y * blockDim.y + threadIdx.y;
  if (idn >= nbatch || idm >= ncomb || idm == 0)
    return;
  // comb[idn, idm], merged[idn], comb_bit[idn, idm]
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
  // comb[idn, idm], merged[idn]
  squant::get_comb_SD_cuda(&comb[idn * ncomb * bra_len + idm * bra_len],
                           &merged[idn * sorb], idm - 1, sorb, noA, noB);
}

// get all Singles-Doubles for given onv(2D)
__host__ void squant::get_comb_cuda(unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  // comb: (nbatch, ncomb, bra_len) merged: (nbatch, sorb)
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_comb_SD_kernel<<<gridDim, blockDim>>>(comb, merged_ovlst, sorb, bra_len,
                                            noA, noB, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

// get all Singles-Doubles and states(3D: nbatch, ncomb, sorb) for given onv(2D)
__host__ void squant::get_comb_cuda(double *comb_bit, unsigned long *comb,
                                    const int *merged_ovlst, const int sorb,
                                    const int bra_len, const int noA,
                                    const int noB, const int nbatch,
                                    const int ncomb) {
  // comb: (nbatch, ncomb, bra_len), comb_bit: (nbatch, ncomb, sorb)
  // merged: [nbatch, sorb]
  dim3 blockDim(THREAD, THREAD);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x,
               (ncomb + blockDim.y - 1) / blockDim.y);
  get_comb_SD_kernel<<<gridDim, blockDim>>>(comb, comb_bit, merged_ovlst, sorb,
                                            bra_len, noA, noB, nbatch, ncomb);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void permuate_sgn_kernel(const int64_t *image2,
                                    const int64_t *onstate, int64_t *index,
                                    int64_t *sgn, const int size,
                                    const size_t nbatch) {
  int idm = blockIdx.x * blockDim.x + threadIdx.x;
  if (idm >= nbatch)
    return;
  sgn[idm] = squant::permute_sgn_cuda(image2, &onstate[idm * size],
                                      &index[idm * size], size);
}

__host__ void squant::permute_sng_batch_cuda(const int64_t *image2,
                                             const int64_t *onstate,
                                             int64_t *index, int64_t *sgn,
                                             const int size,
                                             const int64_t nbatch) {
  dim3 blockDim(1024);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  permuate_sgn_kernel<<<gridDim, blockDim>>>(image2, onstate, index, sgn, size,
                                             nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void array_index_kernel(double *data_ptr, int64_t *index,
                                   int64_t length, int64_t offset,
                                   double **ptr_array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  ptr_array[idn] = data_ptr + index[idn] + offset;
  // printf("idn: %ld, index %ld, ptr_array: %p\n", idn, index[idn], (void
  // *)ptr_array[idn]);
}

__host__ void array_index_cuda(double *data_ptr, int64_t *index, int64_t length,
                               int64_t offset, double **ptr_array) {
  dim3 blockDim(1024);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  array_index_kernel<<<gridDim, blockDim>>>(data_ptr, index, length, offset,
                                            ptr_array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void get_array_kernel(double *data_ptr, int64_t *index,
                                 int64_t length, int64_t offset,
                                 double *array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  array[idn] = *(data_ptr + index[idn] + offset);
}

__host__ void get_array_cuda(double *data_ptr, int64_t *index, int64_t length,
                             int64_t offset, double *array) {
  dim3 blockDim(1024);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  get_array_kernel<<<gridDim, blockDim>>>(data_ptr, index, length, offset,
                                          array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void ones_array_kernel(double *data_ptr, int64_t length,
                                  int64_t stride, int64_t offset) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;
  data_ptr[idn * stride + offset] = 1.0;
}

__host__ void ones_array_cuda(double *data_ptr, int64_t length, int64_t stride,
                              int64_t offset) {
  dim3 blockDim(1024);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  ones_array_kernel<<<gridDim, blockDim>>>(data_ptr, length, stride, offset);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void print_ptr_ptr_kernel(double *data_ptr, double **ptr_array,
                                     int64_t *dr, int64_t *dc, size_t nbatch) {
  for (size_t i = 0; i < nbatch; i++) {
    printf("%ld-th:(%li, %li) \n", i, dr[i], dc[i]);
    for (size_t j = 0; j < dr[i]; j++) {
      for (size_t k = 0; k < dc[i]; k++) {
        printf("%f ", ptr_array[i][j * dc[i] + k]);
      }
      printf(" \n");
    }
  }
}

__global__ void print_ptr_ptr_kernel(double *data_ptr, double **ptr_array,
                                     int64_t *dr, size_t nbatch) {
  for (size_t i = 0; i < nbatch; i++) {
    printf("%ld-th:%li \n", i, dr[i]);
    for (size_t j = 0; j < dr[i]; j++) {
      printf("%f ", ptr_array[i][j]);
    }
    printf(" \n");
  }
}

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, int64_t *dc, size_t nbatch) {
  print_ptr_ptr_kernel<<<1, 1>>>(data_ptr, ptr_array, dr, dc, nbatch);
}

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, size_t nbatch) {
  print_ptr_ptr_kernel<<<1, 1>>>(data_ptr, ptr_array, dr, nbatch);
}

__global__ void swap_pointers_kernel(double **ptr_ptr, double **ptr_ptr_1) {
  double **tmp = ptr_ptr_1;
  tmp = ptr_ptr_1;
  ptr_ptr_1 = ptr_ptr;
  ptr_ptr = tmp;
  tmp = nullptr;
}

__host__ void swap_pointers_cuda(double **ptr_ptr, double **ptr_ptr_1) {
  swap_pointers_kernel<<<1, 1>>>(ptr_ptr, ptr_ptr_1);
}

__global__ void convert_sites_kernel(
    const int64_t *onstate, const int nphysical, const int64_t *data_index,
    const int64_t *qrow_qcol, const int64_t *qrow_qcol_index,
    const int64_t *qrow_qcol_shape, const int64_t *ista,
    const int64_t *ista_index, const int64_t *image2, const int64_t nbatch,
    int64_t *data_info, bool *sym_array) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= nbatch)
    return;
  // onstate: [nbatch, nphysical * 2]
  // data_info [nbatch, nphysical, 3]
  squant::sites_sym_index(&onstate[idn * nphysical * 2], nphysical, data_index,
                          qrow_qcol, qrow_qcol_index, qrow_qcol_shape, ista,
                          ista_index, image2, nbatch, &data_info[idn],
                          &sym_array[idn]);
}

__host__ void convert_sites_cuda(const int64_t *onstate, const int nphysical,
                                 const int64_t *data_index,
                                 const int64_t *qrow_qcol,
                                 const int64_t *qrow_qcol_index,
                                 const int64_t *qrow_qcol_shape,
                                 const int64_t *ista, const int64_t *ista_index,
                                 const int64_t *image2, const int64_t nbatch,
                                 int64_t *data_info, bool *sym_array) {
  // XXX: how to allocate blockDim???, register overflow if blockDim = 1024
  dim3 blockDim(256);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  convert_sites_kernel<<<gridDim, blockDim>>>(
      onstate, nphysical, data_index, qrow_qcol, qrow_qcol_index,
      qrow_qcol_shape, ista, ista_index, image2, nbatch, data_info, sym_array);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}

__global__ void merge_idx_kernel(int64_t *merge_counts, const int64_t *idx,
                                 const int64_t *counts, const int64_t batch) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= batch)
    return;
  // Notice: atomicadd
  merge_counts[idx[idn]] += counts[idn];
}

__host__ void merge_idx_cuda(int64_t *merge_counts, const int64_t *idx,
                             const int64_t *counts, const int64_t batch) {
  dim3 blockDim(1024);
  dim3 gridDim((batch + blockDim.x - 1) / blockDim.x);
  merge_idx_kernel<<<gridDim, blockDim>>>(merge_counts, idx, counts, batch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();
}

__device__ void constrain_charts(const int64_t key, double *result) {
  // {10, 6, 14, 9, 5, 13, 11, 7, 15};
  if (key == 10) {
    result[0] = 1;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 6) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 1;
    result[3] = 0;
  } else if (key == 14) {
    result[0] = 1;
    result[1] = 0;
    result[2] = 1;
    result[3] = 0;
  } else if (key == 9) {
    result[0] = 0;
    result[1] = 1;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 5) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 1;
  } else if (key == 13) {
    result[0] = 0;
    result[1] = 1;
    result[2] = 0;
    result[3] = 1;
  } else if (key == 11) {
    result[0] = 1;
    result[1] = 1;
    result[2] = 0;
    result[3] = 0;
  } else if (key == 7) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 1;
    result[3] = 1;
  } else if (key == 15) {
    result[0] = 1;
    result[1] = 1;
    result[2] = 1;
    result[3] = 1;
  }
}

__global__ void constrain_lookup_table_kernel(const int64_t *sym_index,
                                              double *result,
                                              const int64_t nbatch) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= nbatch)
    return;
  constrain_charts(sym_index[idn], &result[idn * 4]);
}

__host__ void constrain_lookup_table(const int64_t *sym_index, double *result,
                                     const int64_t nbatch) {
  dim3 blockDim(1024);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  constrain_lookup_table_kernel<<<gridDim, blockDim>>>(sym_index, result,
                                                       nbatch);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();
}

template <typename IntType>
__device__ inline int compare(const IntType *mid_element, const IntType *target,
                              const int64_t target_length) {
  for (int64_t i = target_length - 1; i >= 0; i--) {
    if (mid_element[i] < target[i]) {
      return -1;
    } else if (mid_element[i] > target[i]) {
      return 1;
    }
  }
  return 0;
}

template <typename IntType>
__device__ int64_t BigInteger_device(const IntType *arr, const IntType *target,
                                     const int64_t arr_length,
                                     const int64_t target_length = 1,
                                     bool little_endian = true) {
  // arr: [arr_length, targe_length] 2D array but arr is point not point-point
  // arr is array of the great uint64 or others [12, 13] => 2**64 + 12
  // target: [targe_length]
  // little_endian: [12, 13] => 13 * 2**64 + 12
  // big_endian: [12, 13] => 12 * 2**64 + 12
  int64_t left = 0;
  int64_t right = arr_length - 1;

  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    int64_t mid_index = mid * target_length;
    const IntType *mid_element = &arr[mid_index];
    int result = compare(mid_element, target, target_length);

    if (result == 0) {
      return mid;
    } else if (result < 0) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return -1;
}

__global__ void BigInteger_kernel(const unsigned long *arr,
                                  const unsigned long *target, int64_t *result,
                                  bool *mask,
                                  const int64_t nbatch,
                                  const int64_t arr_length,
                                  const int64_t target_length,
                                  bool little_endian) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  // int tx = threadIdx.x;
  // __shared__ unsigned long tmp[256][MAX_SORB_LEN];
  if (idn >= nbatch)
    return;
// #pragma unroll
//   for (int i = 0; i < MAX_SORB_LEN; i++) {
//     tmp[tx][i] = target[target_length * idn + i];
//   }
//   __syncthreads();
//   // result[idn]
  int64_t x = BigInteger_device<unsigned long>(arr, &target[idn * target_length], arr_length,
                                               target_length, little_endian);
  result[idn] = x;
  if (x == -1) {
    mask[idn] = false;
  }
};

__host__ void binary_search_BigInteger_cuda(
    const unsigned long *arr, const unsigned long *target, int64_t *result, 
    bool *mask,
    const int64_t nbatch, const int64_t arr_length,
    const int64_t target_length = 1, bool little_endian = true) {
  dim3 blockDim(256);
  dim3 gridDim((nbatch + blockDim.x - 1) / blockDim.x);
  BigInteger_kernel<<<gridDim, blockDim>>>(
      arr, target, result,mask, nbatch, arr_length, target_length, little_endian);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
}
