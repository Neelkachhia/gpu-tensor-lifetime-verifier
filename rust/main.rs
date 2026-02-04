mod tensor;
use tensor::Tensor;

extern "C"
{
  fn cuda_alloc_and_free();
}

fn main()
{
 let t = Tensor::<f32>::new(128);
 println!("Allocated tensor of length {}", t.len());
}

