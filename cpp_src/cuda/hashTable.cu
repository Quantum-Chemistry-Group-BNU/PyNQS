#include "cuda_handle_error.h"
#include "hashTable_cuda.h"
#include <iomanip>
#include <iostream>

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
      // printf("keyIdx is %d, hashed value is %d, now size is %d, bucket-size
      // %d, error\n", i, hashed_value, write_off, bucket_size); bucket_size
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
      int hv = hashFunc2<MAX_SORB_LEN>(my_value, sizeof(BFT) * 8);
      my_bf |= (1 << hv);
      hv = hashFunc3<MAX_SORB_LEN>(my_value, sizeof(BFT) * 8);
      my_bf |= (1 << hv);
    }
    ht.bf[bid] = my_bf;
  }
  return;
}

template<int _len>
__global__ void hash_lookup_kerenl(myHashTable ht, unsigned long *keys,
                                   int64_t *values, bool *mask,
                                   const int64_t length) {
  int64_t idn = blockIdx.x * blockDim.x + threadIdx.x;
  if (idn >= length)
    return;

  int64_t big_id[_len];
  int64_t off;
    // big_id[0] = keys[3 * idn];
    // big_id[1] = keys[3 * idn + 1];
    // big_id[2] = keys[3 * idn + 2];
    // KeyT key(big_id[0], big_id[1], big_id[2]);
    // off = ht.search_key(key);
  if constexpr (_len == 1) {
    big_id[0] = keys[idn];
    KeyT key(big_id[0]);
    off = ht.search_key(key);
  } else if constexpr (_len == 2) {
    big_id[0] = keys[2 * idn];
    big_id[1] = keys[2 * idn + 1];
    KeyT key(big_id[0], big_id[1]);
    off = ht.search_key(key);
  } else if constexpr (_len == 3) {
    big_id[0] = keys[3 * idn];
    big_id[1] = keys[3 * idn + 1];
    big_id[2] = keys[3 * idn + 2];
    KeyT key(big_id[0], big_id[1], big_id[2]);
    off = ht.search_key(key);
  }
  if (off != -1) {
    values[idn] = ht.values[off].data[0];
  } else {
    values[idn] = -1;
    mask[idn] = false;
  }
}

void hash_lookup(myHashTable ht, unsigned long *keys, int64_t *values,
                 bool *mask, const int64_t length) {
  cudaEvent_t start, stop;
  float esp_time_gpu;

  dim3 blockDim(256);
  dim3 gridDim((length + blockDim.x - 1) / blockDim.x);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  hash_lookup_kerenl<MAX_SORB_LEN><<<gridDim, blockDim>>>(ht, keys, values, mask, length);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&esp_time_gpu, start, stop);
  printf("Time for lookup_hashtable_kernel is: %f ms\n", esp_time_gpu);
}

bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num,
                     int device_index = 0) {

  ht.bNum = bucket_num;
  ht.bSize = bucket_size;
  int total_size = ht.bNum * ht.bSize;

  bool debug = true;
  if (debug) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_index);
    auto max_memory = deviceProp.totalGlobalMem;
    auto memory = sizeof(KeyT) * (total_size) + sizeof(ValueT) * total_size +
                  sizeof(int) * bucket_num + sizeof(BFT) * bucket_num;
    std::cout << "bnum: " << bucket_num << " bsize: " << bucket_size
              << " ele num: " << ele_num << std::setprecision(5)
              << " Memory: " << memory / std::pow(2.0, 20) << " MiB "
              << "Cost: " << memory / (double)max_memory * 100 << " %"
              << std::endl;
  }

  cudaMalloc((void **)&ht.keys, sizeof(KeyT) * total_size);
  cudaMalloc((void **)&ht.values, sizeof(ValueT) * total_size);
  cudaMalloc((void **)&ht.bCount, sizeof(int) * bucket_num);
  cudaMalloc((void **)&ht.bf, sizeof(BFT) * bucket_num);
  cudaMemset(ht.bCount, 0, sizeof(int) * bucket_num);
  cudaMemset(ht.bf, 0, sizeof(BFT) * bucket_num);

  int *build_failure;
  CUDA_TRY(cudaMalloc((void **)&build_failure, sizeof(int)));
  CUDA_TRY(cudaMemset(build_failure, 0, sizeof(int)));

  // build hash table kernel
  // If we need better performance for this process, we can use multi-split.

  cudaEvent_t start, stop;
  float esp_time_gpu;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int block_size = 256;
  int block_num = 2048;
  build_hashtable_kernel<<<block_num, block_size>>>(ht, all_keys, all_values,
                                                    ele_num, build_failure);
  cudaDeviceSynchronize();
  build_hashtable_bf_kernel<<<block_num, block_size>>>(ht);
  cudaError_t cudaStatus = cudaGetLastError();
  HANDLE_ERROR(cudaStatus);
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&esp_time_gpu, start, stop);
  printf("Time for build_hashtable_kernel is: %f ms\n", esp_time_gpu);

  // build success check
  int *build_flag = new int[1];
  cudaMemcpy(build_flag, build_failure, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  bool return_state = build_flag[0] == 0 ? true : false;
  if (build_flag[0] == 1) {
    cudaFree(ht.keys);
    cudaFree(ht.values);
    cudaFree(ht.bCount);
    cudaFree(ht.bf);
  } else {
    printf("build hash table success\n");
  }
  delete[] build_flag;
  CUDA_TRY(cudaFree(build_failure));
  return return_state;
}