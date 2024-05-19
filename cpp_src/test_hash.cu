#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_TRY(call)                                                         \
  do {                                                                         \
    cudaError_t const status = (call);                                         \
    if (cudaSuccess != status) {                                               \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);    \
    }                                                                          \
  } while (0)

class cudaException : public std::exception {
public:
  cudaException(const char *message) noexcept : message(message) {}

  const char *what() const noexcept override { return message.c_str(); }

private:
  std::string message;
};

// copy from:
// https://github.com/ComputationalRadiationPhysics/raptr/blob/master/src/CUDA_HandleError.hpp
/**
 * Wrapper for CUDA functions. On CUDA error prints error message and exits
 * program.
 * @param err CUDA error object.
 * @param file Part of error message: Name of file in that the error occurred.
 * @param line Part of error message: Line in that the error occurred.
 */
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << file << "(" << line << "): Error: " << cudaGetErrorString(err)
              << std::endl;
    throw cudaException(cudaGetErrorString(err));
  }
}

/**
 * Wrapper macro for HandleError(). Arguments 'file' and 'line' that accord to
 * the place where HANDLE_ERROR() is used are passed to HandleError()
 * automatically.
 */
#define HANDLE_ERROR(err)                                                      \
  { HandleError(err, __FILE__, __LINE__); }

struct KeyT {
  char data[16];
  __device__ __host__ KeyT() {}
  __device__ __host__ KeyT(int64_t v1) {
    int64_t *ptr = static_cast<int64_t *>((void *)data);
    ptr[0] = v1;
    ptr[1] = v1;
  }
  __device__ __host__ KeyT(int64_t v1, int64_t v2) {
    // printf("v1: %ld, v2: %ld\n", v1, v2);
    int64_t *ptr = static_cast<int64_t *>((void *)data);
    ptr[0] = v1;
    ptr[1] = v2;
  }
  __device__ __host__ bool operator==(const KeyT key) {
    int64_t *d1 = (int64_t *)key.data;
    int64_t *d2 = (int64_t *)(key.data + 8);
    int64_t *_d1 = (int64_t *)data;
    int64_t *_d2 = (int64_t *)(data + 8);
    return (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
  }
  __device__ __host__ bool operator<(const KeyT key) const {
    int64_t *d1 = (int64_t *)key.data;
    int64_t *d2 = (int64_t *)(key.data + 8);
    int64_t *_d1 = (int64_t *)data;
    int64_t *_d2 = (int64_t *)(data + 8);
    return (_d1[0] < d1[0]) || (_d1[0] == d1[0] && _d2[0] < d2[0]);
  }
};
struct ValueT {
  int64_t data[1];
};

#define _len 16

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
  for (int i = 0; i < _len; i++)
    hash = (hash ^ values[i]) * p;
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
  BFT *bf; // BloomFilter
  int bNum;
  int bSize;
  __inline__ __device__ __host__ int64_t search_key(KeyT key) {
    // printf("key %ld, bNum: %ld", &key, bNum);
    int hashvalue = myHashFunc(key, bNum);
    int my_bucket_size = bCount[hashvalue];
    KeyT *list = keys + (int64_t)hashvalue * bSize;
    int threshold = sizeof(BFT) * 8;
    BFT my_bf = bf[hashvalue];
    // BloomFilter, false positive probabilistic
    if (!((my_bf >> hashFunc2(key, threshold)) & 1) ||
        !((my_bf >> hashFunc3(key, threshold)) & 1)) {
      return -1;
    }
    // printf("hashvalue: %d, bucket-size: %d", hashvalue, bSize);
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
                                       int *build_failure) {
  int bucket_num = ht.bNum;
  int bucket_size = ht.bSize;
  KeyT *keys = ht.keys;
  ValueT *values = ht.values;
  int *bucket_count = ht.bCount;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  for (int i = thread_idx; i < ele_num; i = i + total_threads) {
    KeyT my_key = all_keys[i];
    ValueT my_value = all_values[i];
    int hashed_value = myHashFunc(my_key, bucket_num);
    int write_off = atomicAdd(bucket_count + hashed_value, 1);
    if (write_off >= bucket_size) {
      build_failure[0] = 1;
      // printf("keyIdx is %d, hashed value is %d, now size is %d, bucket-size %d, error\n", i, hashed_value, write_off, bucket_size); 
      // bucket_size
      break;
    }
    keys[hashed_value * bucket_size + write_off] = my_key;
    values[hashed_value * bucket_size + write_off] = my_value;
  }
  return;
}

__global__ void build_hashtable_bf_kernel(myHashTable ht) {
  int bucket_num = ht.bNum;
  int bucket_size = ht.bSize;
  KeyT *keys = ht.keys;
  int *bucket_count = ht.bCount;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int bid = thread_idx; bid < bucket_num; bid += gridDim.x * blockDim.x) {
    int my_bsize = bucket_count[bid];
    BFT my_bf = 0;
    for (int e = 0; e < my_bsize; e++) {
      KeyT my_value = keys[bid * bucket_size + e];
      int hv = hashFunc2(my_value, sizeof(BFT) * 8);
      my_bf |= (1 << hv);
      hv = hashFunc3(my_value, sizeof(BFT) * 8);
      my_bf |= (1 << hv);
    }
    ht.bf[bid] = my_bf;
  }
  return;
}

__global__ void hash_lookup_kerenl(myHashTable ht, unsigned long *keys,
                                   int64_t *values, bool *mask,
                                   const int64_t length) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;

  uint64_t big_id[2];
  big_id[0] = keys[2 * idn];
  big_id[1] = keys[2 * idn + 1];
  KeyT key(big_id[0], big_id[1]);
  int64_t off = ht.search_key(key);

  if (off != -1) {
    values[idn] = ht.values[off].data[0];
  } else {
    values[idn] = -1;
    mask[idn] = false;
  }
  // printf("offset: %ld, values: %ld\n", off, values[idn]);
}

void hash_lookup(myHashTable ht, unsigned long *keys, int64_t *values,
                 bool *mask,
                 const int64_t length) {
  // printf("Lookup length: %ld\n", length);
  cudaEvent_t start, stop;
  float esp_time_gpu;

  dim3 blockDim(256);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  hash_lookup_kerenl<<<gridDim, blockDim>>>(ht, keys, values, mask, length);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);

  (cudaEventRecord(stop, 0));
  (cudaEventSynchronize(stop));
  (cudaEventElapsedTime(&esp_time_gpu, start, stop));
  printf("Time for lookup_kernel is: %f ms\n", esp_time_gpu);
}

bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num) {

  ht.bNum = bucket_num;
  ht.bSize = bucket_size;

  printf("bnum is %d, bsize is %d, ele num is %d\n", bucket_num, bucket_size,
         ele_num);

  int total_size = ht.bNum * ht.bSize;
  //. total memory:
  auto memory = sizeof(KeyT) * (total_size) + sizeof(ValueT) * total_size +
                sizeof(int) * bucket_num + sizeof(BFT) * bucket_num;

    int device_index = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_index);
    auto max_memory = deviceProp.totalGlobalMem;
    std::cout << "Memory: " << memory / std::pow(2.0, 20) << " MiB "
              << "Cost: " << memory / (double)max_memory * 100 << " %" << std::endl;

  std::cout << "Memory: " << memory / std::pow(2.0, 20) << " MiB " << std::endl;
  CUDA_TRY(cudaMalloc((void **)&ht.keys, sizeof(KeyT) * total_size));
  CUDA_TRY(cudaMalloc((void **)&ht.values, sizeof(ValueT) * total_size));
  CUDA_TRY(cudaMalloc((void **)&ht.bCount, sizeof(int) * bucket_num));
  CUDA_TRY(cudaMalloc((void **)&ht.bf, sizeof(BFT) * bucket_num));
  CUDA_TRY(cudaMemset(ht.bCount, 0, sizeof(int) * bucket_num));
  CUDA_TRY(cudaMemset(ht.bf, 0, sizeof(BFT) * bucket_num));

  int *build_failure;
  CUDA_TRY(cudaMalloc((void **)&build_failure, sizeof(int)));
  CUDA_TRY(cudaMemset(build_failure, 0, sizeof(int)));

  // build hash table kernel
  // If we need better performance for this process, we can use multi-split.

  cudaEvent_t start, stop;
  float esp_time_gpu;
  CUDA_TRY(cudaEventCreate(&start));
  CUDA_TRY(cudaEventCreate(&stop));
  CUDA_TRY(cudaEventRecord(start, 0));

  int block_size = 256;
  int block_num = 2048;
  build_hashtable_kernel<<<block_num, block_size>>>(ht, all_keys, all_values,
                                                    ele_num, build_failure);
  CUDA_TRY(cudaDeviceSynchronize());
  build_hashtable_bf_kernel<<<block_num, block_size>>>(ht);
  CUDA_TRY(cudaDeviceSynchronize());

  CUDA_TRY(cudaEventRecord(stop, 0));
  CUDA_TRY(cudaEventSynchronize(stop));
  CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
  printf("Time for build_hashtable_kernel is: %f ms\n", esp_time_gpu);

  // build success check
  int *build_flag = new int[1];
  CUDA_TRY(cudaMemcpy(build_flag, build_failure, sizeof(int),
                      cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaDeviceSynchronize());
  bool return_state = build_flag[0] == 0 ? true : false;
  if (build_flag[0] == 1) {
    CUDA_TRY(cudaFree(ht.keys));
    CUDA_TRY(cudaFree(ht.values));
    CUDA_TRY(cudaFree(ht.bCount));
    CUDA_TRY(cudaFree(ht.bf));
  } else {
    printf("build hash table success\n");
  }
  delete[] build_flag;
  CUDA_TRY(cudaFree(build_failure));
  return return_state;
}

int main() {
  //  /opt/nvidia/bin/nvcc -L/opt/cuda/lib -lcudart -std=c++17 test_hash.cu -O3 && ./a.out

  for(int k = 0; k < 1; k++){
  const int ele_num = 1<< 20;
  std::vector<unsigned long> key(ele_num * 2, 0);

  std::vector<unsigned long> value(ele_num);
  std::iota(value.begin(), value.end(), 0);

  const bool random = true;
  if (random) {
    std::random_device rd;
    // std::mt19937_64 rng(rd());
    std::mt19937 rng(2212);
    std::uniform_int_distribution<int64_t> u0(0, 1 << 31);
    for (int i = 0; i < ele_num; ++i) {
      key[i * 2] = u0(rng);
    }

    std::vector<unsigned long> key_copy;
    std::copy(key.begin(), key.end(), std::back_inserter(key_copy));
    std::sort(key_copy.begin(), key_copy.end());
    auto x = std::unique(key_copy.begin(), key_copy.end());
    auto len = std::distance(key_copy.begin(), x);
    std::cout << "unique length: " << len << std::endl;
 
  } else {
    std::vector<unsigned long> _key(ele_num);
    std::iota(_key.begin(), _key.end(), 0);
    // std::mt19937 rng(2000);
    // std::shuffle(_key.begin(), _key.end(), rng);
    for (int64_t i = 0; i < ele_num; i++) {
      key[i * 2] = _key[i];
    }
  }

  // for (auto &i : value) {
  //   std::cout << i << "\n";
  // }
  // std::cout << std::endl;

  float avg2cacheline = 0.3;
  float avg2bsize = 0.55;
  int cacheline_size = 128 / sizeof(KeyT);
  int avg_size = cacheline_size * avg2cacheline;
  int bucket_size = avg_size / avg2bsize;
  int bucket_num = (ele_num + avg_size - 1) / avg_size;

  myHashTable ht;
  // unsigned long *dev_data = nullptr;
  unsigned long *key_ptr = nullptr;
  unsigned long *value_ptr = nullptr;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "  Total Global Memory (bytes): " << deviceProp.totalGlobalMem << std::endl;

  cudaMalloc((void **)&key_ptr, sizeof(unsigned long) * ele_num * 2);
  cudaMalloc((void **)&value_ptr, sizeof(unsigned long) * ele_num);
  cudaMemcpy(key_ptr, key.data(), sizeof(unsigned long) * ele_num * 2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(value_ptr, value.data(), sizeof(unsigned long) * ele_num,
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  while (!build_hashtable(ht, (KeyT *)key_ptr, (ValueT *)value_ptr, bucket_num,
                          bucket_size, ele_num)) {
    bucket_size = 1.4 * bucket_size;
    avg2bsize = (float)avg_size / bucket_size;
    printf(
        "Build hash table failed! The avg2bsize is %f now. Rebuilding... ...\n",
        avg2bsize);
  }

  // hashlookup-test
  std::vector<unsigned long> key1(ele_num * 2);
  std::vector<int64_t> value1(ele_num);
  std::copy(key.begin(), key.end(), key1.begin());

  unsigned long *key1_ptr = nullptr;
  int64_t *value1_ptr = nullptr;
  cudaMalloc((void **)&key1_ptr, sizeof(unsigned long) * ele_num * 2);
  cudaMalloc((void **)&value1_ptr, sizeof(int64_t) * ele_num);

  // std::cout << "key: " << key1_ptr << " value: " << value1_ptr << std::endl;

  cudaMemcpy(key1_ptr, key1.data(), sizeof(unsigned long) * ele_num * 2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(value1_ptr, value1.data(), sizeof(int64_t) * ele_num,
             cudaMemcpyHostToDevice);
  
  bool mask[ele_num];
  for(int64_t i = 0; i < ele_num; i++){
    mask[i] = true;
  }
  bool *mask_ptr = nullptr;
  cudaMalloc((void **)&mask_ptr, sizeof(bool) * ele_num);
  cudaMemcpy(mask_ptr, mask, sizeof(bool) * ele_num, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  hash_lookup(ht, key1_ptr, value1_ptr, mask_ptr, ele_num);
  cudaMemcpy(value1.data(), value1_ptr, sizeof(int64_t) * ele_num,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(mask, mask_ptr, sizeof(bool) * ele_num,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for(size_t i = 0 ; i < ele_num; i++){
    if(value1[i] != i && mask[i]){
      std::cout << "False" << std::endl;
    }
  }
  std::cout << std::endl;
  freeHashTable(ht);
  }
  return 0;
}