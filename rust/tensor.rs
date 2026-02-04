use std::marker::PhantomData;
use std::ptr::NonNull;

extern "C"
{
  fn cuda_alloc(bytes: usize) -> *mut std::ffi::c_void;
  fn cuda_free(ptr: *mut std::ffi::c_void);
}

pub struct Tensor<T>
{
  ptr: NonNull<T>,
  len: usize,
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

    Tensor
    {
      ptr,
      len,
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
}

impl <T> Drop for Tensor<T>
{
  fn drop(&mut self)
  {
    unsafe
    {
      cuda_free(self.ptr.as_ptr() as *mut std::ffi::c_void);
    }
  }
}