#pragma once
#ifndef CUDA_HANDLE_ERROR
#define CUDA_HANDLE_ERROR

#include <cuda_runtime.h>
#include <iostream>

class cudaException : public std::exception {
 public:
  cudaException(const char* message) noexcept : message(message) {}

  const char* what() const noexcept override { return message.c_str(); }

 private:
  std::string message;
};

// copy from: https://github.com/ComputationalRadiationPhysics/raptr/blob/master/src/CUDA_HandleError.hpp
/**
 * Wrapper for CUDA functions. On CUDA error prints error message and exits program.
 * @param err CUDA error object.
 * @param file Part of error message: Name of file in that the error occurred.
 * @param line Part of error message: Line in that the error occurred.
 */
static void HandleError( cudaError_t err, const char * file, int line )
{
  if(err != cudaSuccess)
    {
      std::cerr << file << "(" << line << "): Error: " << cudaGetErrorString( err ) << std::endl;
      throw cudaException(cudaGetErrorString(err));
    }
}

/**
 * Wrapper macro for HandleError(). Arguments 'file' and 'line' that accord to
 * the place where HANDLE_ERROR() is used are passed to HandleError()
 * automatically.
 */
#define HANDLE_ERROR(err) { HandleError( err, __FILE__, __LINE__ ); }

#endif  // #ifndef CUDA_HANDLE_ERROR
