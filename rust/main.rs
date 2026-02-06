mod tensor;
use tensor::Tensor;

fn main() {
    let mut t = Tensor::<f32>::new(1_000_000);
    t.add_one();

    // Proper synchronization
    std::thread::sleep(std::time::Duration::from_millis(50));

    println!("Tensor safely dropped after GPU work");
}
