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

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <half/half.hpp>
#include <memory>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/TensorImpl.hpp"
#include "mllm/Utils/SymExpr/Eval.hpp"

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

  [[nodiscard]] inline std::shared_ptr<TensorViewImpl> impl() const { return impl_; }

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
    fmt::println("address   :{:#010x}", (uintptr_t)(impl_->address()));
    fmt::println("name      :{}", name().empty() ? "<empty>" : name());
    fmt::println("shape     :{}", fmt::join(shape(), "x"));
    fmt::println("device    :{}", deviceTypes2Str(device()));
    fmt::println("dtype     :{}", dataTypes2Str(dtype()));

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

class TiledTensor {
 public:
  explicit TiledTensor(Tensor& t) : t_(t) {}

  // This `complexLoops` methods is single threads. Do not use this function for things that needs
  // highly paralleled. Use `parallelLoops` instead.
  template<typename T>
  void complexLoops(const std::vector<std::string>& symbol_str,
                    const std::function<void(T* ptr, const std::vector<int32_t>&)>& callback,
                    int32_t top_loop_parallel = 0) {
    std::vector<SymExpr> symbols;
    symbols.reserve(symbol_str.size());
    for (const auto& s : symbol_str) { symbols.emplace_back(s); }

    const int loops = symbols.size();
    const int t_shape_size = t_.shape().size();
    std::unordered_map<std::string, float> loop_symbols_v;
    std::vector<int32_t> offsets(t_.shape().size(), 0);
    std::vector<int32_t> co(t_.shape().size(), 0);

    for (int i = 0; i < t_shape_size; ++i) loop_symbols_v["_" + std::to_string(i)] = 0.f;

    const auto& shape = t_.impl()->shape();

    const int symbol_cnt = symbols.size();
    while (true) {
      // feed back to loop_symbols_v
      for (int i = 0; i < t_shape_size; ++i) {
        loop_symbols_v["_" + std::to_string(i)] = (float)(co[i]);
      }
      for (int i = 0; i < loops; ++i) { offsets[i] = symbols[i].evalAsInt(loop_symbols_v); }

      callback(t_.offsettedPtr<T>(offsets), offsets);

      // package offsets
      int dim = symbol_cnt - 1;
      bool carry = true;
      while (carry && dim >= 0) {
        co[dim] += 1;
        if (co[dim] < shape[dim]) {
          carry = false;
        } else {
          co[dim] = 0;
          --dim;
        }
      }
      if (carry) break;
    }
  }

  template<typename T>
  void parallelLoops(int last_dim,
                     const std::function<void(T* ptr, int b_stride,
                                              const std::vector<int32_t>& left_dims)>& callback,
                     int32_t top_loop_parallel = 0) {
    // check if last dim is contigious. using stride and shape to check
    const int loops = t_.shape().size();
    MLLM_RT_ASSERT(last_dim < loops)
    const auto& this_stride = t_.impl()->stride();
    const auto& this_shape = t_.impl()->shape();
    int contigious_cnt_stride = 1;
    bool contigious_last_dim = true;
    for (int i = loops - 1; i > last_dim; --i) {
      if (contigious_cnt_stride != this_stride[i]) contigious_last_dim = false;
      contigious_cnt_stride *= this_shape[i - 1];
    }
    MLLM_RT_ASSERT(contigious_last_dim);

    std::vector<int32_t> callback_left_dims;
    for (int i = last_dim + 1; i < loops; ++i) { callback_left_dims.emplace_back(this_shape[i]); }

    int parallel_for_loops = this_shape[last_dim];
    std::vector<int32_t> parallel_for_offsets(loops, 0);

#pragma omp parallel for num_threads(top_loop_parallel) schedule(auto) if (top_loop_parallel > 0)
    for (int pl = 0; pl < parallel_for_loops; ++pl) {
      parallel_for_offsets[last_dim] = pl;
      callback(t_.offsettedPtr<T>(parallel_for_offsets), this_stride[last_dim], callback_left_dims);
    }
  }

 private:
  Tensor& t_;
};

// extern template instance. reduce compile time and binary size.
#define EXTERN_TEMPLATE_TILED_TENSOR_DEF(__mllm_type)                                  \
  extern template void TiledTensor::complexLoops<__mllm_type>(                         \
      const std::vector<std::string>&,                                                 \
      const std::function<void(__mllm_type*, const std::vector<int32_t>&)>&, int32_t); \
  extern template void TiledTensor::parallelLoops<__mllm_type>(                        \
      int, const std::function<void(__mllm_type*, int, const std::vector<int32_t>&)>&, int32_t);

EXTERN_TEMPLATE_TILED_TENSOR_DEF(float)
EXTERN_TEMPLATE_TILED_TENSOR_DEF(half_float::half)
EXTERN_TEMPLATE_TILED_TENSOR_DEF(int8_t)
EXTERN_TEMPLATE_TILED_TENSOR_DEF(int16_t)

#define EXTERN_TEMPLATE_TILED_TENSOR_IMPL(__mllm_type)                                 \
  template void TiledTensor::complexLoops<__mllm_type>(                                \
      const std::vector<std::string>&,                                                 \
      const std::function<void(__mllm_type*, const std::vector<int32_t>&)>&, int32_t); \
  template void TiledTensor::parallelLoops<__mllm_type>(                               \
      int, const std::function<void(__mllm_type*, int, const std::vector<int32_t>&)>&, int32_t);

#undef EXTERN_TEMPLATE_TILED_TENSOR_DEF

}  // namespace mllm