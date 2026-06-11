#ifdef GPU

#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "cuda_tensor.h"

template <typename Km>
void swap_pointers(Km **&ptr1, Km **&ptr2) {
  Km **temp = ptr1;
  ptr1 = ptr2;
  ptr2 = temp;
  temp = nullptr;
}

// cpu version
template <typename T>
void array_index_cpu(T *data_ptr, int64_t *index, int64_t length,
                     int64_t offset, T **ptr_array) {
  for (size_t i = 0; i < length; i++) {
    ptr_array[i] = data_ptr + index[i] + offset;
  }
}

template <typename T>
void get_array_cpu(T *data_ptr, int64_t *index, int64_t length, int64_t offset,
                   T *array) {
  for (size_t i = 0; i < length; i++) {
    array[i] = *(data_ptr + index[i] + offset);
  }
}

template <typename T>
void ones_array_cpu(T *data_ptr, int64_t length, int64_t stride) {
  for (size_t i = 0; i < length; i++) {
    data_ptr[i * stride] = static_cast<T>(1);
  }
}

template <typename T>
void print_tensor(Tensor tensor, size_t length, std::string str = "Tensor") {
  std::cout << str << " ";
  auto tmp_tensor = tensor.to(torch::kCPU);
  auto tmp_ptr = tmp_tensor.data_ptr<T>();
  for (size_t i = 0; i < length; i++) {
    std::cout << tmp_ptr[i] << " \n"[i == length - 1];
  }
}

template <typename T, typename K>
void print_ptr_ptr(T *data_ptr, T **ptr_array, K *dr, K *dc, size_t nbatch,
                   std::string str = "ptr-ptr", bool use_cpu = true) {
  std::cout << str << std::endl;
  if (use_cpu) {
    for (size_t i = 0; i < nbatch; i++) {
      std::cout << i << "-batch:"
                << "(" << dr[i] << ", " << dc[i] << ")\n";
      for (size_t j = 0; j < dr[i]; j++) {
        for (size_t k = 0; k < dc[i]; k++) {
          std::cout << ptr_array[i][j * dc[i] + k] << " \n"[k == dc[i] - 1];
        }
      }
    }
  } else {
    print_ptr_ptr_cuda(data_ptr, ptr_array, dr, dc, nbatch);
  }
}

template <typename T, typename K>
void print_ptr_ptr(T *data_ptr, T **ptr_array, K *dr, size_t nbatch,
                   std::string str = "ptr-ptr", bool use_cpu = true) {
  std::cout << str << std::endl;
  if (use_cpu) {
    for (size_t i = 0; i < nbatch; i++) {
      std::cout << i << "-th:" << dr[i] << "\n";
      for (size_t j = 0; j < dr[i]; j++) {
        std::cout << ptr_array[i][j] << " \n"[j == dr[i] - 1];
      }
    }
  } else {
    // K *dr_cpu = new K[nbatch];
    // T **ptr_array_cpu = new T* [nbatch];
    // auto kind = cudaMemcpyDeviceToHost;
    // cudaMemcpy(dr_cpu, dr, sizeof(K) * nbatch, kind);
    // cudaMemcpy(ptr_array_cpu, data_ptr, sizeof(T *) * nbatch, kind);
    print_ptr_ptr_cuda(data_ptr, ptr_array, dr, nbatch);
    // delete[] dr_cpu;
    // delete [] ptr_array_cpu;
  }
}

#endif