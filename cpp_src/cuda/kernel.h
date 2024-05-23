#pragma once
#include <sys/types.h>

#include <cstdint>
#include "utils_cuda.h"
#include "../common/default.h"
#include "utils_cuda.h"
#include <type_traits>

namespace squant {

// states: (nbatch, bra_len * 8), bra: (nbatch, sorb)
__host__ void tensor_to_onv_cuda(uint8_t *states, const uint8_t *bra,
                                 const int sorb, const int nbatch,
                                 const int bra_len, const int tensor_len);

// bra: (nbatch, onv), comb: (nbatch, sorb)
__host__ void onv_to_tensor_cuda(double *comb, const unsigned long *bra,
                                 const int sorb, const int bra_len,
                                 const int nbatch, const size_t numel);

// merge olst and vlst, bra: (nbatch, onv)
__host__ void get_merged_cuda(const unsigned long *bra, int *merged,
                              const int sorb, const int nele, const int bra_len,
                              const int nbatch);

// <i|H|j> i: 2D(nbatch, onv), j: 3D(nbatch, ncomb, onv)
// local energy -> (nbatch, ncomb)
__host__ void get_Hij_3D_cuda(double *Hmat, const unsigned long *bra,
                              const unsigned long *ket, const double *h1e,
                              const double *h2e, const int sorb, const int nele,
                              const int bra_len, const int nbatch,
                              const int ncomb);

// <i|H|j> matrix, i,j: 2D (n, onv), (m, onv)
// construct Hij matrix -> (n, m)
__host__ void get_Hij_2D_cuda(double *Hmat, const unsigned long *bra,
                              const unsigned long *ket, const double *h1e,
                              const double *h2e, const int sorb, const int nele,
                              const int bra_len, const int n, const int m);

// comb_bit: (nbatch, ncomb, sorb)
// comb: (nbatch, ncomb, onv), merged_ovlst: (nbatch, sorb)
__host__ void get_comb_cuda(double *comb_bit, unsigned long *comb,
                            const int *merged_ovlst, const int sorb,
                            const int bra_len, const int noA, const int noB,
                            const int nbatch, const int ncomb);

// comb: (nbatch, ncomb, onv), merged_ovlst: (nbatch, sorb)
__host__ void get_comb_cuda(unsigned long *comb, const int *merged_ovlst,
                            const int sorb, const int bra_len, const int noA,
                            const int noB, const int nbatch, const int ncomb);

__host__ void permute_sng_batch_cuda(const int64_t *image2,
                                     const int64_t *onstate, int64_t *index,
                                     int64_t *sgn, const int size,
                                     const int64_t nbatch);

}  // namespace squant

__host__ void convert_sites_cuda(const int64_t *onstate, const int nphysical,
                                 const int64_t *data_index,
                                 const int64_t *qrow_qcol,
                                 const int64_t *qrow_qcol_index,
                                 const int64_t *qrow_qcol_shape,
                                 const int64_t *ista, const int64_t *ista_index,
                                 const int64_t *image2, const int64_t nbatch,
                                 int64_t *data_info, bool *sym_array);

__host__ void array_index_cuda(double *data_ptr, int64_t *index, int64_t length,
                               int64_t offset, double **ptr_array);

__host__ void get_array_cuda(double *data_ptr, int64_t *index, int64_t length,
                             int64_t offset, double *array);

__host__ void ones_array_cuda(double *data_ptr, int64_t length, int64_t stride,
                              int64_t offset = 0);

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, int64_t *dc, size_t nbatch);

__host__ void print_ptr_ptr_cuda(double *data_ptr, double **ptr_array,
                                 int64_t *dr, size_t nbatch);

__host__ void swap_pointers_cuda(double **ptr_ptr, double **ptr_ptr_1);

__host__ void merge_idx_cuda(int64_t *merge_counts, const int64_t *idx,
                             const int64_t *counts, const int64_t batch);

__host__ void constrain_lookup_table(const int64_t *sym_index, double *result,
                                     const int64_t nbatch);

__host__ void binary_search_BigInteger_cuda(
    const unsigned long *arr, const unsigned long *target, int64_t *result,
    bool *mask, const int64_t nbatch, const int64_t arr_length,
    const int64_t target_length, bool little_endian);

struct KeyT {
  char data[MAX_SORB_LEN * 8];
  __device__ __host__ KeyT() {}

  template <typename... Args>
  __device__ __host__ KeyT(Args... values) {
    static_assert(std::conjunction_v<std::is_same<Args, uint64_t>...>,
                  "Arguments must be is int64_t");
    static_assert(sizeof...(Args) == MAX_SORB_LEN & sizeof...(Args) <= 3,
                  "Arguments error for KeyT constructor");

    uint64_t *ptr = reinterpret_cast<uint64_t *>((void *)data);
    size_t index = 0;
    ((ptr[index++] = values), ...);
  }
  // template <std::size_t N>
  // __device__ __host__ bool compareKeys(const KeyT key1, const KeyT key2) {
  //   int64_t *d = (int64_t *)(key1.data + (N - 1) * 8);
  //   int64_t *_d = (int64_t *)(key2.data + (N - 1) * 8);
  //   return d[0] == _d[0] && compareKeys<N - 1>(key1, key2);
  // }

  // __device__ __host__ bool operator==(const KeyT key){
  //   return compareKeys<MAX_SORB_LEN>(*this, key);
  // }

  // __device__ __host__ bool operator==(const KeyT &key) {
  //   int64_t *d1 = (int64_t *)key.data;
  //   int64_t *_d1 = (int64_t *)data;
  //   return d1[0] == _d1[0];

  __device__ __host__ bool operator==(const KeyT key) {
    uint64_t *d1 = (uint64_t *)key.data;
    uint64_t *_d1 = (uint64_t *)data;
    bool flag = d1[0] == _d1[0];
    if constexpr (MAX_SORB_LEN == 1) {
      ;
    } else if constexpr (MAX_SORB_LEN == 2) {
      uint64_t *d2 = (uint64_t *)(key.data + 8);
      uint64_t *_d2 = (uint64_t *)(data + 8);
      flag &= (d2[0] == _d2[0]);
    }else if constexpr(MAX_SORB_LEN == 3) {
      uint64_t *d2 = (uint64_t *)(key.data + 8);
      uint64_t *_d2 = (uint64_t *)(data + 8);
      flag &= (d2[0] == _d2[0]);
      uint64_t *d3 = (uint64_t *)(key.data + 16);
      uint64_t *_d3 = (uint64_t *)(data + 16);
      flag &= (d3[0] == _d3[0]);
    }
    return flag;
  }
  // __device__ __host__ bool operator<(const KeyT key) const {
  //   // TODO:
  //   int64_t *d1 = (int64_t *)key.data;
  //   int64_t *_d1 = (int64_t *)data;
  //   int64_t *d2 = (int64_t *)(key.data + 8);
  //   int64_t *_d2 = (int64_t *)(data + 8);
  //   return (_d1[0] < d1[0]) || (_d1[0] == d1[0] && _d2[0] < d2[0]);
  // }
};

// explicit specialization in non-namespace scope
// template <>
// inline __device__ __host__ bool KeyT::compareKeys<0>(const KeyT key1,
//                                                      const KeyT key2) {
//   return true;
// }

struct ValueT {
  int64_t data[1];
};

#define CUDA_TRY(call)                                                      \
  do {                                                                      \
    cudaError_t const status = (call);                                      \
    if (cudaSuccess != status) {                                            \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
    }                                                                       \
  } while (0)

__inline__ __device__ __host__ int myHashFunc(KeyT value, int threshold) {
  // BKDR hash
  unsigned int seed = 31;
  char *values = static_cast<char *>(value.data);
  int len = sizeof(KeyT);
  unsigned int hash = 171;
  while (len--) {
    char v = (~values[len - 1]) * (len & 1) + (values[len - 1]) * (~(len & 1));
    hash = hash * seed + (v & 0xF);
  }
  // printf("BKDR hash: %d threshold %d\n", hash & 0x7FFFFFFF, threshold);
  return (hash & 0x7FFFFFFF) % threshold;
}

template <int _len>
__inline__ __device__ __host__ int hashFunc1(KeyT value, int threshold) {
  int p = 16777619;
  int hash = (int)216161L;
  char *values = static_cast<char *>(value.data);
#pragma unroll
  for (int i = 0; i < _len * 8; i++) hash = (hash ^ values[i]) * p;
  hash += hash << 13;
  hash ^= hash >> 7;
  hash += hash << 3;
  hash ^= hash >> 17;
  hash += hash << 5;
  return (hash & 0x7FFFFFFF) % threshold;
}

template <int _len>
__inline__ __device__ __host__ int hashFunc2(KeyT value, int threshold) {
  /*int len = sizeof(KeyT);
  char *values = static_cast<char*>(value.data);
  int hash = 324223113;
  for (int i = 0; i < len; i ++)
      hash = (hash<<4)^(hash>>28)^values[i];
  return (hash & 0x7FFFFFFF) % threshold;*/

  unsigned int seed = 12313;
  char *values = static_cast<char *>(value.data);
  // int _len = sizeof(KeyT);
  unsigned int hash = 711371;
#pragma unroll
  for (int i = _len * 8; i > 0; i--) {
    char v = (~values[i - 1]) * (i & 1) + (values[i - 1]) * (~(i & 1));
    hash = hash * seed + (v & 0xF);
  }
  return (hash & 0x7FFFFFFF) % threshold;
}

template <int _len>
__inline__ __device__ __host__ int hashFunc3(KeyT value, int threshold) {
  // RS hash
  char *values = static_cast<char *>(value.data);
  int b = 378551;
  int a = 63689;
  int hash = 0;
#pragma unroll
  for (int i = 0; i < _len * 8; i++) {
    hash = hash * a + values[i];
    a = a * b;
  }
  return (hash & 0x7FFFFFFF) % threshold;
}

#define BFT uint32_t
struct myHashTable {
  KeyT *keys;
  ValueT *values;
  int *bCount;
  BFT *bf;  // BloomFilter
  int bNum;
  int bSize;
  __inline__ __device__ __host__ int64_t search_key(KeyT key) {
    int hashvalue = myHashFunc(key, bNum);
    int my_bucket_size = bCount[hashvalue];
    KeyT *list = keys + (int64_t)hashvalue * bSize;
    int threshold = sizeof(BFT) * 8;
    BFT my_bf = bf[hashvalue];
    // BloomFilter, false positive probabilistic
    if (!((my_bf >> hashFunc2<MAX_SORB_LEN>(key, threshold)) & 1) ||
        !((my_bf >> hashFunc3<MAX_SORB_LEN>(key, threshold)) & 1)) {
      return -1;
    }
    // printf("hashvalue: %d, bucket-size: %d\n", hashvalue, my_bucket_size);
    // FIXME:(zbwu-05-22`) why is pretty lower, Global memory
    for (int i = 0; i < my_bucket_size; i++) {
      if (list[i] == key) {
        // printf("off: %d", hashvalue * bSize + i);
        return hashvalue * bSize + i;
      }
    }
    return -1;
  }
};

inline void freeHashTable(myHashTable ht) {
  (cudaFree(ht.keys));
  (cudaFree(ht.values));
  (cudaFree(ht.bCount));
  (cudaFree(ht.bf));
}

__host__ bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num,
                     int device_index);

__host__ void hash_lookup(myHashTable ht, unsigned long *keys, int64_t *values,
                 bool *mask, const int64_t length);
