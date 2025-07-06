/**
 * @file BaseOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/Tensor.hpp"
#include "mllm/Engine/ParameterReader.hpp"

namespace mllm {

enum class OpType : int32_t {  // NOLINT
  kOpType_Start = 0,

  kFill,

  kAdd,
  kSub,
  kMul,
  kDiv,

  kMatMul,
  kLLMEmbeddingToken,

  kLinear,
  kRoPE,
  kSoftmax,
  kTranspose,
  kRMSNorm,
  kSiLU,
  kKVCache,
  kCausalMask,
  kCastType,

  kD2H,
  kH2D,

  kSplit,
  kView,

  kFlashAttention_2,

  kRepeat,
  kPermute,

  kOpType_End,
};

inline const char* opType2Str(OpType type) {
  switch (type) {
    case OpType::kOpType_Start: return "kOpType_Start";
    case OpType::kFill: return "kFill";
    case OpType::kAdd: return "kAdd";
    case OpType::kSub: return "kSub";
    case OpType::kMul: return "kMul";
    case OpType::kDiv: return "kDiv";
    case OpType::kMatMul: return "kMatMul";
    case OpType::kLLMEmbeddingToken: return "kLLMEmbeddingToken";
    case OpType::kLinear: return "kLinear";
    case OpType::kRoPE: return "kRoPE";
    case OpType::kSoftmax: return "kSoftmax";
    case OpType::kTranspose: return "kTranspose";
    case OpType::kRMSNorm: return "kRMSNorm";
    case OpType::kSiLU: return "kSiLU";
    case OpType::kKVCache: return "kKVCache";
    case OpType::kCausalMask: return "kCausalMask";
    case OpType::kCastType: return "kCastType";
    case OpType::kD2H: return "kD2H";
    case OpType::kH2D: return "kH2D";
    case OpType::kSplit: return "kSplit";
    case OpType::kView: return "kView";
    case OpType::kFlashAttention_2: return "kFlashAttention_2";
    case OpType::kRepeat: return "kRepeat";
    case OpType::kOpType_End: return "kOpType_End";
    default: return "Unknown";
  }
}

// Keep the CRTP Design Pattern Pls.
template<typename DerivedT>
class BaseOpCargo {
 public:
  BaseOpCargo() = default;
  BaseOpCargo(size_t inputs_len, size_t outputs_len, DataTypes default_dtype = kFp32)
      : inputs_dtypes_(inputs_len, default_dtype), outputs_dtypes_(outputs_len, default_dtype) {}

  DerivedT& setInputsDtype(size_t pos, DataTypes dtype) {
    if (pos >= inputs_dtypes_.size()) { inputs_dtypes_.resize(pos + 1, kFp32); }
    inputs_dtypes_[pos] = dtype;
    return *static_cast<DerivedT*>(this);
  }

  DerivedT& setOutputsDtype(size_t pos, DataTypes dtype) {
    if (pos >= outputs_dtypes_.size()) { outputs_dtypes_.resize(pos + 1, kFp32); }
    outputs_dtypes_[pos] = dtype;
    return *static_cast<DerivedT*>(this);
  }

  [[nodiscard]] int thread() const { return threads_; }

  void setThreads(int threads) { threads_ = threads; }

 private:
  int threads_ = 4;
  std::vector<DataTypes> inputs_dtypes_;
  std::vector<DataTypes> outputs_dtypes_;
};

// type erase wrapper
class BaseOpCargoBase {
 public:
  // do not mark this explicit
  template<typename T>
  BaseOpCargoBase(const T& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<T>>(cargo)) {}

  // do not mark this explicit
  template<typename T>
  BaseOpCargoBase(T&& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(cargo))) {}

  template<typename T>
  const T& as() const {
    if (auto p = dynamic_cast<const Model<T>*>(inner_.get())) { return p->data_; }
    throw std::bad_cast();
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
  };

  template<typename T>
  struct Model : Concept {
    Model(const T& data) : data_(data) {}  // NOLINT(google-explicit-constructor)
    T data_;
  };

  std::unique_ptr<Concept> inner_;
};

class BaseOp : public std::enable_shared_from_this<BaseOp> {
 public:
  using params_t = std::unordered_map<std::string, Tensor>;

  explicit BaseOp(OpType op_type);

  virtual void load(const std::shared_ptr<ParameterLoader>& ploader) {};

  virtual void trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {};

  virtual void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

  virtual void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

  virtual void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

  virtual params_t params() { return {}; }

  [[nodiscard]] std::string name() const;

  void setName(const std::string& name);

  [[nodiscard]] DeviceTypes device() const;

  void setDeviceType(DeviceTypes device_type);

 private:
  DeviceTypes device_type_;
  std::string name_;
  OpType op_type_;
};

struct BaseOpFactory {
  virtual ~BaseOpFactory() = default;
  virtual std::shared_ptr<BaseOp> create(const BaseOpCargoBase& base_cargo) = 0;
  [[nodiscard]] virtual OpType opType() const = 0;
};

template<OpType type, typename CargoT>
class TypedOpFactory : public BaseOpFactory {
 public:
  std::shared_ptr<BaseOp> create(const BaseOpCargoBase& base_cargo) override {
    const auto& cargo = base_cargo.as<CargoT>();
    return createOpImpl(cargo);
  }

  [[nodiscard]] OpType opType() const override { return type; }

 protected:
  virtual std::shared_ptr<BaseOp> createOpImpl(const CargoT& cargo) = 0;
};

}  // namespace mllm
