mod tensor;
use tensor::Tensor;

fn main()
{
 let mut t = Tensor::<f32>::new(128);
 t.record_use();
 println!("Tensor recorded GPU use");
}

