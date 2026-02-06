# GPU Tensor Lifetime Verifier

A **Rust + CUDA runtime prototype** that detects **asynchronous GPU use-after-free bugs** by verifying tensor lifetimes with **CUDA events**.

This project demonstrates how Rust ownership and RAII can be extended beyond CPU memory to reason about **GPU execution timelines**, a problem faced by deep learning frameworks, GPU runtimes, and driver-level systems.

---

## 🚀 Why This Project Exists

GPU kernels execute **asynchronously** with respect to the CPU. This makes memory safety extremely hard:

* The CPU may free GPU memory **while a kernel is still running**
* Rust’s borrow checker cannot see GPU execution
* Bugs manifest as **silent corruption or crashes**

This project builds a **runtime verifier** that:

* Tracks when a GPU tensor was last used
* Associates that use with a **CUDA event**
* Prevents deallocation until the GPU has finished execution

If a tensor is dropped too early, the program **fails loudly and deterministically**.

---

## 🧠 Core Idea (One Sentence)

> A GPU tensor is safe to free **only after the CUDA event recorded after its last kernel launch has completed**.

---

## 🏗️ Architecture Overview

```
Rust (Host)
 └── Tensor<T>
     ├── Owns GPU memory (cudaMalloc / cudaFree)
     ├── Tracks last CUDA event
     ├── Records event after kernel launch
     └── Verifies event completion on Drop

C / CUDA (Device Runtime)
 ├── Memory allocation helpers
 ├── CUDA event management
 └── Asynchronous CUDA kernels
```

Rust never talks to CUDA directly. All interactions go through a **C-compatible CUDA wrapper**, keeping the unsafe boundary explicit and minimal.

---

## 📁 Project Structure

```
gpu-tensor-lifetime-verifier/
├── cuda/
│   └── cuda_api.cu        # CUDA memory, events, and kernels
├── rust/
│   ├── main.rs            # Demo + bug trigger
│   └── tensor.rs          # Tensor abstraction + lifetime verifier
└── libcudawrap.so         # CUDA shared library (built locally)
```

---

## 🔒 Safety Model

### What Is Guaranteed

* GPU memory is freed **exactly once**
* No tensor can be dropped while the GPU is still using it
* Async use-after-free bugs are detected at runtime

### What Is Not Attempted

* Compile-time GPU lifetime proofs
* Multi-GPU or multi-stream correctness
* Performance optimization

This is a **correctness-first runtime prototype**.

---

## ⚙️ How Lifetime Verification Works

1. A CUDA kernel is launched asynchronously
2. A CUDA event is recorded immediately after launch
3. The event is stored inside the owning `Tensor`
4. When the tensor is dropped:

   * The runtime queries the event
   * If the GPU has not finished → **panic**

This mirrors how real GPU runtimes reason about execution progress.

---

## 🧪 Demonstrated Bug Detection

The demo intentionally triggers a real GPU bug:

```rust
{
    let mut t = Tensor::<f32>::new(1_000_000);
    t.add_one();          // async kernel launch
} // tensor dropped too early
```

### Runtime Output

```
Kernel launched, tensor will now go out of scope
thread 'main' panicked at 'Tensor dropped while GPU is still using it!'
```

This confirms the verifier correctly detects **async GPU use-after-free**.

---

## 🛠️ Build & Run

### Prerequisites

* Linux (tested on WSL2)
* NVIDIA GPU + CUDA Toolkit
* Rust toolchain

### Build CUDA Library

```bash
nvcc -Xcompiler -fPIC -shared cuda/cuda_api.cu -o libcudawrap.so
```

### Build & Run Rust Demo

```bash
cd rust
rustc main.rs -L .. -l cudawrap -o main
LD_LIBRARY_PATH=.. ./main
```

---

## 📌 Key Learnings

* GPU memory safety cannot be reasoned about without execution timelines
* Rust ownership can be extended to **foreign, asynchronous systems**
* CUDA events are the minimal primitive needed for correctness
* Runtime verification is often the only practical solution

---

## 🎯 Who This Is For

* GPU runtime / driver engineers
* Systems programmers working near hardware
* Deep learning framework developers
* Compiler and runtime researchers

---

## 🔮 Possible Extensions

* Stream-aware lifetime tracking
* Multi-tensor dependency graphs
* Deterministic execution modes
* Cargo + build.rs integration
* Static + runtime hybrid verification

---

## ⚠️ Disclaimer

This project is a **research and learning prototype**. It is not intended for production use, but demonstrates core ideas used in real GPU runtimes and deep learning systems.
