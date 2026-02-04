#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
  void cuda_alloc_and_free()
  {
    float *d_data = nullptr;
    cudaError_t err = cudaMalloc(&d_data, 10 * sizeof(float));
    if(err != cudaSuccess)
    {
      printf("cudaMalloc failed\n");
      return;
    }
    cudaFree(d_data);
    printf("CUDA allocation + free successful\n");
  }
}