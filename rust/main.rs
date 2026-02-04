extern "C"
{
  fn cuda_alloc_and_free();
}

fn main()
{
  unsafe
  {
    cuda_alloc_and_free();
  }
  println!("Rust successfully called CUDA!");
}