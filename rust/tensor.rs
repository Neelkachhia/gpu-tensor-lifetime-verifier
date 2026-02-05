use std::marker::PhantomData;
use std::ptr::NonNull;
use std::ffi::c_void;

extern "C"
{
  fn cuda_alloc(bytes: usize) -> *mut c_void;
  fn cuda_free(ptr: *mut c_void);
}

#[repr(C)]
struct CudaEvent
{
  _private: [u8; 0],
}

extern "C"
{
 fn cuda_event_create() -> *mut CudaEvent;
 fn cuda_event_record(e: *mut CudaEvent);
 fn cuda_event_query(e: *mut CudaEvent) -> i32;
 fn cuda_event_destroy(e: *mut CudaEvent);
}

pub struct Tensor<T>
{
  ptr: NonNull<T>,
  len: usize,
  last_event: Option<NonNull<CudaEvent>>,
  _marker: PhantomData<T>,
}

impl<T> Tensor<T>
{
  pub fn new(len: usize) -> Self
  {
    let bytes = len * std::mem::size_of::<T>();

    let ptr = unsafe
    {
      cuda_alloc(bytes)
    };

    let ptr = NonNull::new(ptr as *mut T)
    .expect("CUDA allocation failed");

    let event = unsafe
    {
      cuda_event_create()
    };

    let event = NonNull::new(event)
    .expect("Failed to create CUDA event");

    Tensor
    {
      ptr,
      len,
      last_event: Some(event),
      _marker: PhantomData,
    }
  }

  pub fn len(&self) -> usize
  {
    self.len
  }

  pub fn as_ptr(&self) -> *mut T
  {
    self.ptr.as_ptr()
  }

  pub fn record_use(&mut self)
  {
    if let Some(event) = self.last_event
    {
      unsafe
      {
        cuda_event_record(event.as_ptr());
      }
    }
  }
}

impl <T> Drop for Tensor<T>
{
  fn drop(&mut self)
  {
    if let Some(event) = self.last_event
    {
      let done = unsafe
      {
        cuda_event_query(event.as_ptr())
      };

      if done == 0
      {
        panic!("Tensor dropped while GPU is still using it!");
      }
      unsafe
      {
        cuda_event_destroy(event.as_ptr());
      }
    }

    unsafe
    {
      cuda_free(self.ptr.as_ptr() as *mut std::ffi::c_void);
    }
  }
}