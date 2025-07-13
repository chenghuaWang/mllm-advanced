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

/**
 * @class Tensor
 * @brief Represents a multi-dimensional array (tensor) with various operations and metadata.
 *
 * This class provides functionality for tensor creation, manipulation, and inspection.
 * It supports operations like arithmetic, slicing, reshaping, device transfer, and more.
 * Tensors can reside on different devices (CPU/GPU/NPU) and support multiple data types.
 */
class Tensor {
 public:
  using shape_t = TensorViewImpl::shape_t;
  using dtype_t = TensorViewImpl::dtype_t;
  using device_t = TensorViewImpl::device_t;

  /**
   * @brief Default constructor. Creates an empty (null) tensor.
   */
  Tensor() = default;

  /**
   * @brief If this tensor is not initialized
   *
   * @return true
   * @return false
   */
  [[nodiscard]] inline bool isNil() const { return impl_ == nullptr; }

  /**
   * @brief  Create a nil tensor
   *
   * @return Tensor
   */
  static inline Tensor nil() { return {}; }

  /**
   * @brief Constructs a tensor from an existing TensorViewImpl.
   * @param impl Shared pointer to the underlying implementation object.
   */
  explicit Tensor(const std::shared_ptr<TensorViewImpl>& impl);

  /**
   * @brief Creates an uninitialized tensor with specified shape and attributes.
   * @note Empty tensor also has its TensorStorageImpl. Which means it is unique.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFp32).
   * @param device Target device (default: kCPU).
   * @return New tensor but NO MEMORY ALLOCTED!!!
   */
  static Tensor empty(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  /**
   * @brief Allocates memory for the tensor if not already allocated.
   * @return Reference to this tensor for chaining.
   * @note Must be called before accessing data in uninitialized tensors.
   */
  Tensor& alloc();

  /**
   * @brief Creates and attaches an auxiliary tensor view to this tensor.
   * @param extra_tensor_name Unique identifier for the auxiliary view.
   * @param shape Dimensions of the auxiliary tensor.
   * @param dtype Data type (default: kFp32).
   * @param device Target device (default: kCPU).
   * @return Reference to this tensor for chaining.
   * @note This function is designed for quantized Tensor. If one Tensor is quantized to int8 using
   * per tensor quantization method, you can use this_tensor.allocExtraTensorView("scale", shape,
   * kFp32, kCPU); to attach a `scale` tensor to this tensor.
   */
  Tensor& allocExtraTensorView(const std::string& extra_tensor_name,
                               const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                               DeviceTypes device = kCPU);

  /**
   * @brief Retrieves a previously attached auxiliary tensor view.
   * @param extra_tensor_name Name of the auxiliary tensor.
   * @return The requested tensor view.
   * @note This function is designed for quantized Tensor. If one Tensor is quantized to int8 using
   * per tensor quantization method, you can use
   * this_tensor.getExtraTensorViewInTensor("scale").item<float>(); to get a `scale` tensor from
   * this tensor.
   */
  Tensor getExtraTensorViewInTensor(const std::string& extra_tensor_name);

  /**
   * @brief Creates a tensor filled with zeros.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFp32).
   * @param device Target device (default: kCPU).
   * @return New tensor with initialized zero values.
   */
  static Tensor zeros(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                      DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor filled with ones.
   * @param shape Dimensions of the tensor.
   * @param dtype Data type (default: kFp32).
   * @param device Target device (default: kCPU).
   * @return New tensor with initialized one values.
   */
  static Tensor ones(const std::vector<int32_t>& shape, DataTypes dtype = kFp32,
                     DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor with evenly spaced values within a specified range.
   * @param start
   * @param end
   * @param step
   * @param dtype
   * @param device
   * @return Tensor
   */
  static Tensor arange(float start, float end, float step, DataTypes dtype = kFp32,
                       DeviceTypes device = kCPU);

  /**
   * @brief Creates a tensor with random values within a specified range.
   * @param shape
   * @param start
   * @param end
   * @param dtype
   * @param device
   * @return Tensor
   */
  static Tensor random(const std::vector<int32_t>& shape, float start = -1.f, float end = 1.f,
                       DataTypes dtype = kFp32, DeviceTypes device = kCPU);

  /**
   * @brief Creates a shallow view (slice) of the tensor.
   * @param slice_index Slice specification.
   * @return New tensor view referencing the sliced data.
   * @note Uses shallow copy when step size is 1; may be unsafe for GPU tensors.
   */
  Tensor operator[](const SliceIndices& slice_index);

  /**
   * @brief Accesses a tensor element at specified coordinates.
   * @tparam T Expected data type (must match tensor dtype).
   * @param offsets Multi-dimensional indices.
   * @return Reference to the element.
   */
  template<typename T>
  T& at(const std::vector<int32_t>& offsets) {
    return *(offsettedPtr<T>(offsets));
  }

  /// @name Arithmetic Operations
  /// Element-wise operations between tensors.
  /// @{
  Tensor operator+(const Tensor& rhs);
  Tensor operator-(const Tensor& rhs);
  Tensor operator*(const Tensor& rhs);
  Tensor operator/(const Tensor& rhs);
  /// @}

  /// @name Scalar Operations
  /// Element-wise operations with scalar values.
  /// @{
  Tensor operator+(float rhs);
  Tensor operator-(float rhs);
  Tensor operator*(float rhs);
  Tensor operator/(float rhs);
  /// @}

  /**
   * @brief Swaps two dimensions of the tensor.
   * @param dim0 First dimension index.
   * @param dim1 Second dimension index.
   * @return New tensor with transposed dimensions.
   */
  Tensor transpose(int dim0, int dim1);

  /**
   * @brief Transfers tensor to specified device.
   * @param device Target device.
   * @return New tensor on target device (data copied if needed).
   */
  Tensor to(DeviceTypes device);

  /**
   * @brief Converts tensor to specified data type.
   * @param dtype Target data type.
   * @return New tensor with converted data type.
   */
  Tensor to(DataTypes dtype);

  /**
   * @brief Shortcut for moving tensor to CPU.
   * @return CPU-resident tensor.
   */
  Tensor cpu();

  /**
   * @brief Shortcut for moving tensor to GPU.
   * @return GPU-resident tensor.
   */
  Tensor cuda();

  /**
   * @brief Gets the tensor's name.
   * @return Name string (empty if unnamed).
   */
  [[nodiscard]] std::string name() const;

  /**
   * @brief Gets memory type.
   * @return Memory type identifier.
   */
  [[nodiscard]] TensorMemTypes memType() const;

  /**
   * @brief Sets tensor name.
   * @param name New name for tensor.
   * @return Reference to this tensor for chaining.
   */
  Tensor& setName(const std::string& name);

  /**
   * @brief Sets memory type.
   * @param mem_type New memory type.
   * @return Reference to this tensor for chaining.
   */
  Tensor& setMemType(TensorMemTypes mem_type);

  /**
   * @brief Gets data type.
   * @return Current data type.
   */
  [[nodiscard]] DataTypes dtype() const;

  /**
   * @brief Gets device location.
   * @return Current device type.
   */
  [[nodiscard]] DeviceTypes device() const;

  /**
   * @brief Gets tensor dimensions.
   * @return Shape vector.
   */
  [[nodiscard]] std::vector<int32_t> shape() const;

  /**
   * @brief Calculates total number of elements.
   * @return Product of all dimensions.
   */
  [[nodiscard]] size_t numel() const;

  /**
   * @brief Gets unique tensor ID.
   * @return Universally unique identifier.
   */
  [[nodiscard]] uint32_t uuid() const;

  /**
   * @brief Checks memory layout contiguity.
   * @return True if memory is contiguous.
   */
  [[nodiscard]] bool isContiguous() const;

  /**
   * @brief Creates contiguous copy if non-contiguous.
   * @return Contiguous tensor (may be a view or copy).
   */
  Tensor contiguous();

  /**
   * @brief Reshapes tensor without changing data order.
   * @param shape New dimensions.
   * @return Reshaped tensor view.
   */
  Tensor reshape(const std::vector<int>& shape);

  /**
   * @brief Experimental: Creates tensor view with custom indexing.
   * @param indicies View specification.
   * @return New tensor view.
   * @warning This function is in an early age.
   */
  Tensor view(const std::vector<int>& indicies);

  /**
   * @brief Repeats tensor along a dimension.
   *
   * @param multiplier
   * @param dim
   * @return Tensor
   */
  Tensor repeat(int32_t multiplier, int32_t dim);

  /**
   * @brief Unsqueeze tensor along a dimension.
   *
   * @param dim
   * @return Tensor
   */
  Tensor unsqueeze(int32_t dim);

  /**
   * @brief Permute tensor to a new shape
   *
   * @param indices
   * @return Tensor
   */
  Tensor permute(const std::vector<int32_t>& indices);

  /**
   * @brief Gets raw pointer offset by indices.
   * @param offsets Multi-dimensional indices.
   * @return Raw char pointer to the memory location.
   */
  char* offsettedRawPtr(const std::vector<int32_t>& offsets);

  /**
   * @brief Accesses the underlying implementation object.
   * @return Shared pointer to TensorViewImpl.
   */
  [[nodiscard]] inline std::shared_ptr<TensorViewImpl> impl() const { return impl_; }

  /**
   * @brief Typed pointer access with offset.
   * @tparam T Expected data type.
   * @param offsets Multi-dimensional indices.
   * @return Typed pointer to the element.
   */
  template<typename T>
  T* offsettedPtr(const std::vector<int32_t>& offsets) {
    return impl_->offsettedPtr<T>(offsets);
  }

  /**
   * @brief Gets base pointer of tensor data.
   * @tparam T Expected data type.
   * @return Typed base pointer.
   */
  template<typename T>
  T* ptr() const {
    return impl_->ptr<T>();
  }

  /**
   * @brief Accesses scalar value (for 0D or 1-element tensors).
   * @tparam T Expected data type.
   * @return Reference to the scalar value.
   */
  template<typename T>
  T& item() {
    return at<T>({0});
  }

  /**
   * @brief Prints tensor metadata and data.
   * @tparam T Data type for interpretation.
   * @details Output includes address, name, shape, device, dtype, and formatted data.
   * @note Data printing is truncated for large tensors.
   */
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

  /**
   * @brief print tensor shape
   *
   */
  inline void printShape() const { fmt::println("shape: {}", fmt::join(shape(), "x")); }

  /**
   * @brief return how many bytes this tensor alloced.
   *
   * @return size_t
   */
  size_t bytes();

 private:
  /**
   * @brief Internal helper for formatted data printing.
   * @tparam T Data type for interpretation.
   * @details Prints tensor data recursively with indentation. Truncates large dimensions.
   */
  template<typename T>
  void printData() {
    const auto& tensor_shape = shape();
    if (tensor_shape.empty()) {
      fmt::println("[]");
      return;
    }
    printRecursive<T>(tensor_shape, 0, {}, "");
  }

  /**
   * @brief Recursively prints tensor data per dimension.
   * @tparam T Data type for interpretation.
   * @param shape Tensor dimensions.
   * @param dim Current dimension being processed.
   * @param indices Accumulated indices for lower dimensions.
   * @param indent Current indentation string.
   * @note Automatically truncates dimensions >32 elements.
   */
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

  ///< Primary tensor data implementation
  std::shared_ptr<TensorViewImpl> impl_ = nullptr;

  ///< Auxiliary tensor views
  std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>> extra_tensor_view_;
};

class Affine {
 public:
  Affine() = delete;
  Affine(const std::string& sym_exp_str, std::unordered_map<std::string, float>& co);

  inline int operator()() { return expr_.evalAsInt(co_); }

 private:
  SymExpr expr_;
  std::unordered_map<std::string, float>& co_;
};

class AffinePrimitives {
 public:
  Affine create(const std::string& sym_exp_str);

  inline float& operator[](const std::string& key) { return co_[key]; }

 private:
  std::unordered_map<std::string, float> co_;
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