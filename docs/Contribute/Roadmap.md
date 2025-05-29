# Roadmap

- Release Manager: TBD
- Code Freeze Date: TBD

## P0

- [ ] **Server:** Move `Models/AutoLLM` to `Server/Generator`
- [ ] **Backend:** Arm, we used the `kleidiai` in an wrong way.
- [x] **Core:** Refactor the current `TensorImpl` into `TensorViewImpl` and `StorageImpl`. @chenghuaWang
    - Replace `stride_offsets` with `storage_offsets`.
    - Enable `Tensor` to support `operator[]` slicing based on `TensorViewImpl`.
- [ ] **Backend:** Arm backend. Migrate and refactor from the mllm main repository to this repository.
- [ ] **Backend:** QNN backend. Migrate and refactor from the mllm main repository to this repository.
- [ ] **Examples:** Port all examples from the mllm main repository.
- [ ] **Tool:** Implement GGML Quantization method.
- [ ] **Core:** Check the index correctness of `view`, `split`, and `complexLoop`.
- [ ] **Core:** MoE support.
- [ ] **Core:** Refactor `DeviceTypes` and `DataTypes` to make it compatible with dlpack.
- [ ] **Backend:** Arm, support Sparse Attention for ViT.
- [ ] **Models:** Qwen2VL series and Qwen2.5-VL series.
- [ ] **Backend:** Arm, Support Sage Attention. (In MobiAttn repo) @chenghuaWang

## P1

- [ ] **Backend:** CUDA Backend. @chenghuaWang
- [ ] **pymllm:** TileLang backend. @chenghuaWang
