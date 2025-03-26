/**
 * @file Tensor.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief The Tensor abstract. Keep in mind that tensor class does not own the data!!!
 * @version 0.1
 * @date 2025-01-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "fmt/base.h"
#include "fmt/ranges.h"
#include <memory>
#include <cstdint>
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"

namespace mllm {

enum SliceIndexPlaceHolder : int32_t {
  kSliceIndexPlaceHolder_Start = 0x7FFFFFF0,
  kAll,  // 0x7FFFFFF1
  kSliceIndexPlaceHolder_End,
};

struct SliceIndicesPair {
  SliceIndicesPair() = default;
  SliceIndicesPair(int32_t v);  // NOLINT(google-explicit-constructor)
  SliceIndicesPair(int32_t start, int32_t end, int32_t step = 1);

  int32_t start_ = kAll;
  int32_t end_ = kAll;
  int32_t step_ = 1;
};

using SliceIndices = std::vector<SliceIndicesPair>;

class Tensor {
 public:
  Tensor() = default;

  explicit Tensor(const std::shared_ptr<TensorViewImpl>& impl);

  static Tensor empty(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  Tensor& alloc();

  static Tensor zeros(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  static Tensor ones(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                     DeviceTypes device = kCPU);

  // Make a slice of the tensor.
  // If step is 1, this slice will always trying to use shallow copy.
  Tensor operator[](const SliceIndices& slice_index);

  // Make a slice of the tensor.
  // ALWAYS Deep Copy.
  Tensor operator()(const SliceIndices& slice_index);

  Tensor operator+(const Tensor& rhs);

  Tensor operator-(const Tensor& rhs);

  Tensor operator*(const Tensor& rhs);

  Tensor operator/(const Tensor& rhs);

  Tensor operator+(float rhs);

  Tensor operator-(float rhs);

  Tensor operator*(float rhs);

  Tensor operator/(float rhs);

  Tensor transpose(int dim0, int dim1);

  Tensor to(DeviceTypes device);

  Tensor to(DataTypes dtype);

  Tensor cpu();

  Tensor cuda();

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  Tensor& setName(const std::string& name);

  Tensor& setMemType(TensorMemTypes mem_type);

  [[nodiscard]] DataTypes dtype() const;

  [[nodiscard]] DeviceTypes device() const;

  [[nodiscard]] std::vector<int32_t> shape() const;

  [[nodiscard]] size_t numel() const;

  [[nodiscard]] uint32_t uuid() const;

  [[nodiscard]] bool isContiguous() const;

  Tensor contiguous();

  Tensor reshape(const std::vector<int>& shape);

  // FIXME: This function is in an early age.
  Tensor view(const std::vector<int>& indicies);

  char* offsettedRawPtr(const std::vector<int32_t>& offsets);

  template<typename T>
  T* offsettedPtr(const std::vector<int32_t>& offsets) {
    return impl_->offsettedPtr<T>(offsets);
  }

  template<typename T>
  T* ptr() const {
    return impl_->ptr<T>();
  }

  template<typename T>
  void print() {
    fmt::println("Tensor Meta Info");
    fmt::println("address:  {:#010x}", (uintptr_t)(impl_->rptr()));
    fmt::println("name:     {}", name().empty() ? "<empty>" : name());
    fmt::println("shape:    {}", fmt::join(shape(), "x"));
    fmt::println("device:   {}", deviceTypes2Str(device()));
    fmt::println("dtype:    {}", dataTypes2Str(dtype()));

    printData<T>();
  }

 private:
  template<typename T>
  void printData() {
    const auto& tensor_shape = shape();
    if (tensor_shape.empty()) {
      fmt::println("[]");
      return;
    }
    printRecursive<T>(tensor_shape, 0, {}, "");
  }

  template<typename T>
  void printRecursive(const std::vector<int32_t>& shape, size_t dim,
                      std::vector<int32_t> indices,  // NOLINT(performance-unnecessary-value-param)
                      const std::string& indent) {
    size_t dim_size = shape[dim];
    bool is_last_dim = (dim == shape.size() - 1);
    const size_t threshold = 32;
    bool truncated = dim_size > threshold;

    std::vector<int64_t> display_indices;
    if (truncated) {
      for (size_t i = 0; i < 6; ++i) display_indices.push_back(i);
      display_indices.push_back(-1);
      for (size_t i = dim_size - 6; i < dim_size; ++i) display_indices.push_back(i);
    } else {
      for (size_t i = 0; i < dim_size; ++i) display_indices.push_back(i);
    }

    fmt::print("[");
    std::string new_indent = indent + "  ";
    bool first = true;

    for (auto idx : display_indices) {
      if (!first) {
        if (is_last_dim) {
          fmt::print(", ");
        } else {
          fmt::print("\n{}", new_indent);
        }
      }
      first = false;

      if (idx == -1) {
        fmt::print("...");
        continue;
      }

      auto current_indices = indices;
      current_indices.push_back(static_cast<size_t>(idx));

      if (is_last_dim) {
        T* ptr = impl_->offsettedPtr<T>(current_indices);
        fmt::print("{}", *ptr);
      } else {
        printRecursive<T>(shape, dim + 1, current_indices, new_indent);
      }
    }

    fmt::print("]");
    if (dim == 0) fmt::println("");
  }

  std::shared_ptr<TensorViewImpl> impl_ = nullptr;
};

}  // namespace mllm