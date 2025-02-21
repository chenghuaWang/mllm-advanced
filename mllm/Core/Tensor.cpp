/**
 * @file Tensor.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/Tensor.hpp"
#include "mllm/Core/AOps/ElewiseOp.hpp"
#include "mllm/Core/AOps/FillOp.hpp"
#include "mllm/Core/AOps/TransposeOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

SliceIndicesPair::SliceIndicesPair(int32_t v) : start_(v), end_(v + 1), step_(1) {
  if (v == kAll) {
    start_ = kAll;
    end_ = kAll;
  }
}

SliceIndicesPair::SliceIndicesPair(int32_t start, int32_t end, int32_t step)
    : start_(start), end_(end), step_(step) {}

Tensor::Tensor(const std::shared_ptr<TensorImpl>& impl) : impl_(impl) {}

Tensor Tensor::empty(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
  return Tensor(impl);
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
  MllmEngineCtx::instance().mem()->alloc(impl);
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 0},
                                            {Tensor(impl)})[0];
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DataTypes dtype, DeviceTypes device) {
  auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
  MllmEngineCtx::instance().mem()->alloc(impl);
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 1},
                                            {Tensor(impl)})[0];
}

Tensor& Tensor::alloc() {
  if (impl_->rptr()) {
    MLLM_WARN("Tensor already allocated. Tensor uuid is <{}> name is <{}>", impl_->uuid(),
              impl_->name());
    return *this;
  }
  MllmEngineCtx::instance().mem()->alloc(impl_);
  return *this;
}

Tensor& Tensor::to(DeviceTypes device) {
  if (device == impl_->device()) { return *this; }
  // TODO create a function like op
  // H2D, D2H Ops
  // TODO reset device
  return *this;
}

Tensor& Tensor::to(DataTypes dtype) {
  if (dtype == impl_->dtype()) { return *this; }
  // TODO create a function like op
  // data casting ops
  // TODO reset dtype
  return *this;
}

Tensor Tensor::operator[](const SliceIndices& slice_index) {
  return refFrom(slice_index).contiguous();
}

Tensor Tensor::operator+(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kAdd, AddOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator-(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kSub, SubOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator*(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kMul, MulOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator/(const Tensor& rhs) {
  return MllmEngineCtx::instance().dispatch(OpType::kDiv, DivOpCargo{}, {*this, rhs})[0];
}

Tensor Tensor::operator+(float rhs) {
  if (dtype() != kFp32) {
    MLLM_ERROR_EXIT(kError, "Trying to add float constant value with non-float tensor");
  }
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  *(st.ptr<float>()) = rhs;
  return *this + st;
}

Tensor Tensor::operator-(float rhs) {
  if (dtype() != kFp32) {
    MLLM_ERROR_EXIT(kError, "Trying to sub float constant value with non-float tensor");
  }
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  *(st.ptr<float>()) = rhs;
  return *this - st;
}

Tensor Tensor::operator*(float rhs) {
  if (dtype() != kFp32) {
    MLLM_ERROR_EXIT(kError, "Trying to mul float constant value with non-float tensor");
  }
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  *(st.ptr<float>()) = rhs;
  return *this * st;
}

Tensor Tensor::operator/(float rhs) {
  if (dtype() != kFp32) {
    MLLM_ERROR_EXIT(kError, "Trying to div float constant value with non-float tensor");
  }
  auto st = Tensor::empty({1}, dtype(), device()).alloc();
  *(st.ptr<float>()) = rhs;
  return *this / st;
}

Tensor Tensor::transpose(int dim0, int dim1) {
  auto shape_size = impl_->shape().size();
  MLLM_RT_ASSERT(dim0 < shape_size);
  MLLM_RT_ASSERT(dim1 < shape_size);

  return MllmEngineCtx::instance().dispatch(
      OpType::kTranspose,
      TransposeOpCargo{.transpose_dim_x = (size_t)dim0, .transpose_dim_y = (size_t)dim1},
      {*this})[0];
}

std::string Tensor::name() const { return impl_->name(); }

TensorMemTypes Tensor::memType() const { return impl_->memType(); }

Tensor& Tensor::setName(const std::string& name) {
  impl_->setName(name);
  return *this;
}

Tensor& Tensor::setMemType(TensorMemTypes mem_type) {
  impl_->setMemType(mem_type);
  return *this;
}

DataTypes Tensor::dtype() const { return impl_->dtype(); }

DeviceTypes Tensor::device() const { return impl_->device(); }

std::vector<size_t> Tensor::shape() const { return impl_->shape(); }

size_t Tensor::numel() const { return impl_->numel(); }

uint32_t Tensor::uuid() const { return impl_->uuid(); }

Tensor Tensor::contiguousRefFrom(const std::vector<size_t>& offsets) {
  if (!isContiguous()) {
    MLLM_ERROR_EXIT(kError, "contiguousRefFrom required the refed tensor is contiguous");
  }

  auto shape = impl_->shape();
  MLLM_RT_ASSERT_EQ(shape.size(), offsets.size());

  // check if legal
  bool legal = true;
  int none_zero_pos = -1;
  bool find_none_zero = false;
  for (int i = 0; i < shape.size(); ++i) {
    // index cannot oob
    if (offsets[i] >= shape[i]) {
      legal = false;
      break;
    }

    if (offsets[i] != 0) {
      if (find_none_zero) {
        legal = false;
        break;
      } else {
        none_zero_pos = i;
        find_none_zero = true;
      }
    }
  }

  MLLM_RT_ASSERT_EQ(legal, true);

  for (int i = 0; i < none_zero_pos; ++i) {
    if (shape[i] != 1) {
      legal = false;
      break;
    }
  }

  MLLM_RT_ASSERT_EQ(legal, true);

  auto new_shape = impl_->shape();
  for (int i = 0; i < new_shape.size(); ++i) { new_shape[i] = shape[i] - offsets[i]; }

  // Note that: This reference function is marked explicitly contiguous. Which means that there is
  // no need to take consider of ret_impl's stride. The default TensorImpl constructor will
  // calculate stride for it.
  auto ref_impl = std::make_shared<TensorImpl>(new_shape, impl_->dtype(), impl_->device());
  auto ptr_offsets = impl_->stride()[none_zero_pos] * offsets[none_zero_pos];
  ref_impl->_setRawPtr((char*)(impl_->rptr())
                       + (size_t)((float)ptr_offsets * dataTypeSize(impl_->dtype())));

  // Note that: Even this tensor is a ref of old tensor. This tensor should have different name and
  // UUID from old tensor.
  ref_impl->setName(impl_->name() + "_ref_" + std::to_string(ref_impl->uuid()));

  // FIXME: if per-channel quantize and the offsets is set at the channel dim. Error will occurred.
  ref_impl->setQuantizePayload(impl_->quantizePayload());

  Tensor ref_tensor(ref_impl);

  // Set memtype to reference. Which means that this tensor will not be freed automatically.
  ref_tensor.setMemType(kReference);

  return ref_tensor;
}

Tensor Tensor::refFrom(const SliceIndices& slice_indices) {
  if (!impl_->isContiguous()) {
    MLLM_ERROR_EXIT(kError, "refFrom only support on contiguous tensor.");
  }

  MLLM_RT_ASSERT_EQ(slice_indices.size(), impl_->shape().size());

  auto old_shape = impl_->shape();
  auto old_stride = impl_->stride();

  std::vector<size_t> new_shape(old_shape.size());
  std::vector<size_t> new_offsets(old_shape.size());

  for (int i = 0; i < slice_indices.size(); ++i) {
    auto& indices = slice_indices[i];

    auto s = indices.start_;
    auto e = indices.end_;
    auto st = indices.step_;

    if (s == kAll && e == kAll) {
      new_shape[i] = old_shape[i];
      new_offsets[i] = 0;
      continue;
    }

    if (s != kAll && e == kAll) {
      new_shape[i] = old_shape[i] - s;
      new_offsets[i] = s;
      continue;
    }

    if (s != kAll && e != kAll) {
      new_shape[i] = e - s;
      new_offsets[i] = s;
      continue;
    }

    if (s == kAll && e != kAll) {
      new_shape[i] = e;
      new_offsets[i] = 0;
    }
  }

  // wrap to ref tensor
  auto ref_impl = std::make_shared<TensorImpl>(new_shape, old_stride, new_offsets, impl_->dtype(),
                                               impl_->device());
  ref_impl->_setRawPtr(impl_->rptr());
  ref_impl->setName(impl_->name() + "_sliceref_" + std::to_string(ref_impl->uuid()));

  // FIXME. if per-channel quantize and the offsets is set at the channel dim. Error will occurred.
  ref_impl->setQuantizePayload(impl_->quantizePayload());

  // Set memtype to reference. Which means that this tensor will not be freed automatically.
  ref_impl->setMemType(kReference);

  return Tensor(ref_impl);
}

bool Tensor::isContiguous() const { return impl_->isContiguous(); }

Tensor Tensor::contiguous() {
  if (isContiguous()) { return *this; }
  return MllmEngineCtx::instance().dispatch(OpType::kFill, FillOpCargo{.type = 5}, {*this})[0];
}

Tensor Tensor::reshape(const std::vector<int>& shape) {
  // TODO
}

Tensor& Tensor::view(const std::vector<int>& indicies) {
  if (!isContiguous()) {
    MLLM_ERROR_EXIT(kError, "Can not view on non-contiguous tensor. Pls use reshape instead.");
  }

  std::vector<int32_t> new_shape;

  int acc = 1;
  for (auto idx : indicies) {
    new_shape.emplace_back(idx);
    acc *= idx;
  }

  MLLM_RT_ASSERT_EQ(acc, impl_->numel());

  // reset shape and stride
  impl_->setShape(new_shape);

  return *this;
}

char* Tensor::offsettedRawPtr(const std::vector<size_t>& offsets) {
  return impl_->offsettedRawPtr(offsets);
}

}  // namespace mllm