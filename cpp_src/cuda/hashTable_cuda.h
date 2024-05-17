#pragma once

#include <cstdint>
#include <cstdio>

#include "../common/default.h"
#include "utils_cuda.h"

#define _len 16
#define MAX_LEN 2

// ref: https://github.com/linhu-nv/unitTestLocalEnergy
// struct KeyT {
//   char data[MAX_LEN * 8] = {0};
//   __device__ __host__ KeyT() {}
//   __device__ __host__ KeyT(unsigned long v1) {
//     unsigned long *ptr = static_cast<unsigned long *>((void *)data);
//     ptr[0] = v1;
//     ptr[1] = v1;
//   }
//   __device__ __host__ KeyT(unsigned long v1, unsigned long v2) {
//     if (MAX_LEN >= 2) {
//       unsigned long *ptr = static_cast<unsigned long *>((void *)data);
//       ptr[0] = v1;
//       ptr[1] = v2;
//     }
//   }

//   __device__ __host__ KeyT(unsigned long v1, unsigned long v2,
//                            unsigned long v3) {
//     if (MAX_LEN >= 3) {
//       unsigned long *ptr = static_cast<unsigned long *>((void *)data);
//       ptr[0] = v1;
//       ptr[1] = v2;
//       ptr[3] = v3;
//     }
//   }

//   __device__ __host__ KeyT(unsigned long *v, int len) {
//     switch (len) {
//       case 1: {
//         (KeyT(v[0]));
//       }
//       case 2: {
//         (KeyT(v[0], v[1]));
//       }
//       case 3: {
//         (KeyT(v[0], v[1], v[2]));
//       }
//     }
//   }

//   __device__ __host__ bool operator==(const KeyT key) {
//     unsigned long *d1 = (unsigned long *)key.data;
//     unsigned long *d2 = (unsigned long *)(key.data + 8);
//     unsigned long *_d1 = (unsigned long *)data;
//     unsigned long *_d2 = (unsigned long *)(data + 8);
//     bool flag = (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
//     if (MAX_LEN == 3) {
//       unsigned long *d3 = (unsigned long *)(key.data + 16);
//       unsigned long *_d3 = (unsigned long *)(data + 16);
//       bool flag2 = (d3[0] == _d3[0] && d3[0] == _d3[0]) ? true : false;
//       flag = flag && flag2;
//     }
//     return flag;
//   }
// };

// struct ValueT {
//   int64_t data[1];
// };

struct KeyT{
    char data[16];
    __device__ __host__ KeyT() {}
    __device__ __host__ KeyT(int64_t v1) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v1;
    }
    __device__ __host__ KeyT(int64_t v1, int64_t v2) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v2;
    }
    __device__ __host__ bool operator == (const KeyT key) {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
    }
    __device__ __host__ bool operator < (const KeyT key) const {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (_d1[0] < d1[0]) ||  (_d1[0] == d1[0] && _d2[0] < d2[0]);
    }
};
struct ValueT{
    int64_t data[1];
};


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

__inline__ __device__ __host__ int hashFunc1(KeyT value, int threshold) {
  int p = 16777619;
  int hash = (int)216161L;
  char *values = static_cast<char *>(value.data);
#pragma unroll
  for (int i = 0; i < _len; i++) hash = (hash ^ values[i]) * p;
  hash += hash << 13;
  hash ^= hash >> 7;
  hash += hash << 3;
  hash ^= hash >> 17;
  hash += hash << 5;
  return (hash & 0x7FFFFFFF) % threshold;
}

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
  for (int i = _len; i > 0; i--) {
    char v = (~values[i - 1]) * (i & 1) + (values[i - 1]) * (~(i & 1));
    hash = hash * seed + (v & 0xF);
  }
  return (hash & 0x7FFFFFFF) % threshold;
}

__inline__ __device__ __host__ int hashFunc3(KeyT value, int threshold) {
  // RS hash
  char *values = static_cast<char *>(value.data);
  int b = 378551;
  int a = 63689;
  int hash = 0;
#pragma unroll
  for (int i = 0; i < _len; i++) {
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
    // int my_bucket_size = bCount[hashvalue];
    // KeyT *list = keys + (int64_t)hashvalue * bSize;
    // int threshold = sizeof(BFT) * 8;
    // BFT my_bf = bf[hashvalue];
    // // BloomFilter, false positive probabilistic
    // if (!((my_bf >> hashFunc2(key, threshold)) & 1) ||
    //     !((my_bf >> hashFunc3(key, threshold)) & 1)) {
    //   return -1;
    // }
    // printf("hashvalue: %d, bucket-size: %d", hashvalue, 1211);
  //   for (int i = 0; i < my_bucket_size; i++) {
  //     if (list[i] == key) {
  //       // printf("off: %d", hashvalue * bSize + i);
  //       return hashvalue * bSize + i;
  //     }
  //   }
  //   return -1;
  }
};

inline void freeHashTable(myHashTable ht) {
  (cudaFree(ht.keys));
  (cudaFree(ht.values));
  (cudaFree(ht.bCount));
  (cudaFree(ht.bf));
}

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

__global__ void build_hashtable_kernel(myHashTable ht, KeyT *all_keys,
                                       ValueT *all_values, int ele_num,
                                       int *build_failure);

__global__ void build_hashtable_bf_kernel(myHashTable ht);

bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num);

void hash_lookup(myHashTable &ht, unsigned long *keys, int64_t *values,
                 const int64_t length);