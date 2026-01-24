#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_one(float* data)
{
  int idx = threadIdx.x;
  data[idx] += 1.0f;
}

int main()
{
  float* d_data;
  cudaMalloc(&d_data, 10 * sizeof(float));
  add_one <<< 1, 10 >>>(d_data);
  cudaDeviceSynchronize();
  cudaFree(d_data);
  printf("Done\n");
  return 0;
}