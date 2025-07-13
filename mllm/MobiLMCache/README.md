# Mobile LM Cache

## QNN Paged Cache

Since Qnn PageCache requires a special `Qnn_Tensor_t`, and to avoid introducing dependencies in `MllmRT`, we have migrated this functionality to `QnnBackend/MobiLMCache`.

Since QNN's ION memory operates using memory descriptors, we cannot manage memory using the `ptr+offset` approach. The QNN Paged Cache is more like a collection of Tensors, with each Page existing independently as a Tensor. Therefore, the size of each Page can be freely adjusted and is not restricted.
