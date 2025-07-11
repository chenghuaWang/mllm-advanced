# MLLM JIT

The MLLM JIT (Just-In-Time) module leverages the Halide framework to generate optimized operators and performs intelligent shape search tailored for specific hardware architectures. Additionally, the artifacts produced by the JIT engine can be serialized into shared libraries for third-party integration.

**Limitations:**

Currently, within the MLLM framework, Halide functions as a kernel generator, but its native execution is restricted to X86 platforms. Notably, while `mllm-jit` itself is platform-specific, the `mllm-jit` module enables cross-compilation of kernels to various other platforms and architectures through its **AOT (Ahead-Of-Time)** compilation mode.  

**Key Design Notes:**

- The JIT engine is architected as a standalone component independent of the MLLM core engine.
- Linking against `libMllmRT` does not automatically include JIT dependencies. This design decision significantly reduces runtime footprint, enhancing deployment flexibility and resource efficiency.
