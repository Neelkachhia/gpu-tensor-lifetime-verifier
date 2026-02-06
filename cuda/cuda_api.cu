#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
  __global__ void add_one_kernel(float* data, int n) 
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
      data[idx] += 1.0f;
  }
    
  void launch_add_one(float* data, int n) 
  {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_one_kernel<<<blocks, threads>>>(data, n);
  }

  void* cuda_alloc(size_t bytes)
  {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
      return nullptr;
    return ptr;
  }
    
  void cuda_free(void* ptr)
  {
    if (ptr) 
      cudaFree(ptr);
  }

  typedef struct
   {
     cudaEvent_t event;
   } CudaEvent;
 
   CudaEvent* cuda_event_create()
   {
     CudaEvent* e = (CudaEvent*)malloc(sizeof(CudaEvent));
     cudaEventCreateWithFlags(&e->event, cudaEventDisableTiming);
     return e;
   }
 
   void cuda_event_record(CudaEvent* e)
   {
     cudaEventRecord(e->event,0); //default stream
   }
 
   int cuda_event_query(CudaEvent* e)
   {
     cudaError_t status = cudaEventQuery(e->event);
     return status == cudaSuccess;
   }
 
   void cuda_event_destroy(CudaEvent* e)
   {
     cudaEventDestroy(e->event);
     free(e);
   }
 }
    

    
