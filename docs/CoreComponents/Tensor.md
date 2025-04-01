# Tensor

```{figure} ../_static/img/tensor-storage.png
:width: 80%
:alt: Overview
:align: center

Figure 1: Tensor Storage and Tensor View.
```

## Memory Layout

In Mllm, the memory layout of a Tensor is described using the following key concepts:

- **Shape:** Represents the dimensions of the tensor as an array of integers. For example, a tensor with shape `[3, 4, 5]` is a three-dimensional tensor with sizes `3`, `4`, and `5` along its respective dimensions.
- **Stride:** Specifies the number of elements to step in each dimension when moving to the next element in memory. Strides define how to navigate through the memory for each dimension. For instance, a 2D tensor with shape `[3, 4]` and strides `[4, 1]` means that moving along the first dimension skips `4` elements, while moving along the second dimension skips just `1` element.
- **Storage Offset:** Indicates the starting offset within the underlying storage where the tensor's data begins. This allows tensors to share the same storage while accessing different sub-regions via distinct offsets.

When calculating a pointer (ptr) to access a specific element in the tensor, we can use the following formula:

```text
ptr = storage.ptr + (storage_offsets + dot_product(indices, stride)) * sizeof(datatype)
```

## Tiled Tensor

The `TiledTensor` class wraps a `Tensor` object to provide efficient iteration and manipulation capabilities, particularly for tensors with complex memory layouts or those requiring parallel processing. It offers two primary loop mechanisms: `complexLoops` for general, potentially non-contiguous access using symbolic indexing, and `parallelLoops` for optimized parallel iteration over contiguous memory regions.

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

## Future Work

1. Support `CuTe`-like tensors, which utilize a hierarchy of indices to represent tensors [1].

**Reference:**

1. [Graphene: An IR for Optimized Tensor Computations on GPUs](https://dl.acm.org/doi/abs/10.1145/3582016.3582018)
