#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
  void* cuda_alloc(size_t bytes)
  {
    void* ptr = nullptr;
    if(cudaMalloc(&ptr, bytes) != cudaSuccess)
      return nullptr;
    return ptr;
  }

  void cuda_free(void* ptr)
  {
    if(ptr)
      cudaFree(ptr);
  }
}