#include "cuda_handle_error.h"
#include "hashTable_cuda.h"
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
      // printf("keyIdx is %d, hashed value is %d, now size is %d, bucket-size %d, error\n", i, hashed_value, write_off, bucket_size);
      // overflow bucket_size
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

bool build_hashtable(myHashTable &ht, KeyT *all_keys, ValueT *all_values,
                     int bucket_num, int bucket_size, int ele_num) {

  // ht.bNum = bucket_num;
  // ht.bSize = bucket_size;
  // // std::cout << "bucket-num: " << bucket_num << " bucket-size: " <<
  // bucket_size
  // //           << " ele-num: " << ele_num << std::endl;
  // int total_size = ht.bNum * ht.bSize;
  ht.bNum = bucket_num;
  ht.bSize = bucket_size;

  printf("bnum is %d, bsize is %d, ele num is %d\n", bucket_num, bucket_size,
         ele_num);

  int total_size = ht.bNum * ht.bSize;
  //. total memory:
  // 占用总内存
  auto memory= sizeof(KeyT)*(total_size) + sizeof(ValueT)*total_size + sizeof(int)*bucket_num + sizeof(BFT)*bucket_num;
  
  std::cout << "Memory: " << memory/std::pow(2.0, 20) << " MiB " << std::endl;
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