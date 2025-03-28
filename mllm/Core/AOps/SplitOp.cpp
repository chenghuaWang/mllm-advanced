/**
 * @file SplitOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/SplitOp.hpp"

namespace mllm {

SplitOp::SplitOp(const SplitOpCargo& cargo) : BaseOp(OpType::kSplit), cargo_(cargo) {}

void SplitOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing.
}

void SplitOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                    std::vector<Tensor>& outputs) {
  NYI("SplitOp::trace");
}

void SplitOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // do nothing. All things will be done in reshape stage. No memory copy is involved in this op.
}

void SplitOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto const& it = inputs[0];
  int split_at_dim = cargo_.dim_;
  std::vector<int> section_sizes;

  if (split_at_dim < 0) split_at_dim = it.shape().size() + split_at_dim;

  if (cargo_.split_size_or_sections_.size() == 1) {
    MLLM_RT_ASSERT_EQ(it.shape()[split_at_dim] % cargo_.split_size_or_sections_[0], 0);

    for (int i = 0; i < it.shape()[split_at_dim] / cargo_.split_size_or_sections_[0]; ++i) {
      section_sizes.push_back(cargo_.split_size_or_sections_[0]);
    }
  } else {
    int cnt = 0;
    for (int split_size_or_section : cargo_.split_size_or_sections_) {
      cnt += split_size_or_section;
    }

    MLLM_RT_ASSERT_EQ(cnt, it.shape()[split_at_dim]);

    for (int split_size_or_section : cargo_.split_size_or_sections_) {
      section_sizes.push_back(split_size_or_section);
    }
  }

  // Ok. We can now start to split the tensor. Pls calculate storage offsets and stride carefully.
  auto orig_storage = it.impl()->storage();
  int32_t orig_storage_offset = it.impl()->storageOffset();
  auto orig_stride = it.impl()->stride();
  auto orig_shape = it.impl()->shape();

  int sum = 0;
  for (int section_size : section_sizes) {
    std::vector<int32_t> new_shape(orig_shape.begin(), orig_shape.end());
    new_shape[split_at_dim] = section_size;

    int32_t new_storage_offset = orig_storage_offset + sum * orig_stride[split_at_dim];

    outputs.emplace_back(
        TensorViewImpl::create(new_storage_offset, new_shape, orig_stride, orig_storage));

    sum += section_size;
  }
}

void SplitOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // do nothing. All things will be done in reshape stage. No memory copy is involved in this op.
}

}  // namespace mllm
