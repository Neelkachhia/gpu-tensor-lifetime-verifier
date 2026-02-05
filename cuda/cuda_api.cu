#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
{
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

    
