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
  kAll,
  kAuto,
  kSliceIndexPlaceHolder_End,
};

class SliceIndex {
  // TODO
};

class Tensor {
 public:
  explicit Tensor(const std::shared_ptr<TensorImpl>& impl);

  // TODO
  // static Tensor randoms();
  // static Tensor zeros();
  static Tensor empty(const std::vector<size_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  Tensor& alloc();

  // static Tensor arange();
  // static Tensor ones();
  // static Tensor eyes();

  // TODO
  // Tensor& operator[](const SliceIndex& slice_index);

  Tensor operator+(const Tensor& rhs);

  Tensor& to(DeviceTypes device);

  Tensor& to(DataTypes dtype);

  [[nodiscard]] std::string name() const;

  [[nodiscard]] TensorMemTypes memType() const;

  Tensor& setName(const std::string& name);

  Tensor& setMemType(TensorMemTypes mem_type);

  [[nodiscard]] DataTypes dtype() const;

  [[nodiscard]] DeviceTypes device() const;

  [[nodiscard]] std::vector<size_t> shape() const;

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
    T* data = impl_->ptr<T>();
    const auto& tensor_shape = shape();
    if (tensor_shape.empty()) {
      fmt::println("[]");
      return;
    }
    printRecursive<T>(data, tensor_shape, 0, 0, "");
  }

  template<typename T>
  void printRecursive(T* data, const std::vector<size_t>& shape, size_t dim, size_t offset,
                      const std::string& indent) {
    size_t dim_size = shape[dim];
    bool is_last_dim = (dim == shape.size() - 1);
    const size_t threshold = 32;
    bool truncated = dim_size > threshold;

    std::vector<long long> indices;
    if (truncated) {
      for (size_t i = 0; i < 6; ++i) indices.push_back(i);
      indices.push_back(-1);
      for (size_t i = dim_size - 6; i < dim_size; ++i) indices.push_back(i);
    } else {
      for (size_t i = 0; i < dim_size; ++i) indices.push_back(i);
    }

    fmt::print("[");
    std::string new_indent = indent + "  ";

    bool first_element = true;
    for (auto idx : indices) {
      if (!first_element) {
        if (is_last_dim) {
          fmt::print(", ");
        } else {
          fmt::print("\n{}", new_indent);
        }
      } else {
        first_element = false;
      }

      if (idx == -1) {
        fmt::print("...");
        continue;
      }

      size_t i = static_cast<size_t>(idx);
      if (is_last_dim) {
        fmt::print("{}", data[offset + i]);
      } else {
        size_t stride = 1;
        for (size_t d = dim + 1; d < shape.size(); ++d) stride *= shape[d];
        size_t new_offset = offset + i * stride;
        printRecursive<T>(data, shape, dim + 1, new_offset, new_indent);
      }
    }

    fmt::print("]");
    if (dim == 0) fmt::println("");
  }

  std::shared_ptr<TensorImpl> impl_ = nullptr;
};

}  // namespace mllm