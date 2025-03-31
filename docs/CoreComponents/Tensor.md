# Tensor

```{figure} ../_static/img/tensor-storage.png
:width: 80%
:alt: Overview
:align: center

Figure 1: Tensor Storage and Tensor View.
```

## Memory Layout


## Affine Primitives

Mllm offers a straightforward affine mechanism to assist in indexing data from tensors. Currently, the affine primitives are based on **Symbolic Expressions**, which use an Abstract Syntax Tree (AST) to represent the computation. However, relying on ASTs is not an ideal choice due to the associated overhead. One might consider using **Expression Template Trees** to construct affine indices during the compile stage as a more efficient alternative.

The affine primitives are specifically designed to provide dynamic shape support for the Mllm IR Graph.

Below is a simple example of how to use the affine primitives:

```c++
AffinePrimitives ap;
auto affine = ap.create("i * 128 + 64");

for (int i = 0; i < 128; ++i) {
    ap["i"] = i;
    MLLM_RT_ASSERT_EQ(i * 128 + 64, affine());
}
```

**Reference:**

1. [Graphene: An IR for Optimized Tensor Computations on GPUs](https://dl.acm.org/doi/abs/10.1145/3582016.3582018)
