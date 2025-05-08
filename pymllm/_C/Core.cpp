/**
 * @file Core.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "pymllm/_C/Core.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

void registerCoreBinding(py::module_& m) {
  py::enum_<mllm::DeviceTypes>(m, "DeviceTypes")
      .value("CPU", mllm::DeviceTypes::kCPU)
      .value("CUDA", mllm::DeviceTypes::kCUDA)
      .value("OpenCL", mllm::DeviceTypes::kOpenCL);

  py::enum_<mllm::OpType>(m, "OpTypes")
      .value("OpType_Start", mllm::OpType::kOpType_Start)
      .value("Fill", mllm::OpType::kFill)
      .value("Add", mllm::OpType::kAdd)
      .value("Sub", mllm::OpType::kSub)
      .value("Mul", mllm::OpType::kMul)
      .value("Div", mllm::OpType::kDiv)
      .value("MatMul", mllm::OpType::kMatMul)
      .value("LLMEmbeddingToken", mllm::OpType::kLLMEmbeddingToken)
      .value("Linear", mllm::OpType::kLinear)
      .value("RoPE", mllm::OpType::kRoPE)
      .value("Softmax", mllm::OpType::kSoftmax)
      .value("Transpose", mllm::OpType::kTranspose)
      .value("RMSNorm", mllm::OpType::kRMSNorm)
      .value("SiLU", mllm::OpType::kSiLU)
      .value("KVCache", mllm::OpType::kKVCache)
      .value("CausalMask", mllm::OpType::kCausalMask)
      .value("CastType", mllm::OpType::kCastType)
      .value("D2H", mllm::OpType::kD2H)
      .value("H2D", mllm::OpType::kH2D)
      .value("Split", mllm::OpType::kSplit)
      .value("View", mllm::OpType::kView)
      .value("FlashAttention_2", mllm::OpType::kFlashAttention_2)
      .value("OpType_End", mllm::OpType::kOpType_End);
}