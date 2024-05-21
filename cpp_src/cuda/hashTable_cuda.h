#pragma once

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../common/default.h"
#include "utils_cuda.h"

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
    // FIXME: why is pretty lower
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

__global__ void build_hashtable_kernel(myHashTable ht, KeyT *all_keys,
                                       ValueT *all_values, int ele_num,
                                       int *build_failure);

__global__ void build_hashtable_bf_kernel(myHashTable ht);

bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num,
                     int device_index);

void hash_lookup(myHashTable ht, unsigned long *keys, int64_t *values,
                 bool *mask, const int64_t length);