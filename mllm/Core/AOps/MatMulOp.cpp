/**
 * @file MatMulOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/MatMulOp.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

MatMulOp::MatMulOp(const MatMulOpCargo& cargo) : BaseOp(OpType::kMatMul), cargo_(cargo) {}

void MatMulOp::load(const std::shared_ptr<ParameterLoader>& ploader) {
  // do nothing
}

void MatMulOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs) {
  auto ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);
  ctx->create<ir::linalg::MatMulOp>(this, i_irs, o_irs);
}

void MatMulOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("MatMulOp::forward is not implemented");
}

void MatMulOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto shape_a = inputs[0].shape();
  auto shape_b = inputs[1].shape();

  std::vector<size_t> shape_c;

  // check.
  auto size_a = shape_a.size();
  auto size_b = shape_b.size();
  for (int i = 0; i < size_a - 2; ++i) { shape_c.push_back(shape_a[i]); }

  // transform shape.
  // MxK, KxN
  if (!cargo_.transpose_a && !cargo_.transpose_b) {
    MLLM_RT_ASSERT_EQ(shape_a[size_a - 1], shape_b[size_b - 2]);
    shape_c.push_back(shape_a[size_a - 2]);
    shape_c.push_back(shape_b[size_b - 1]);
  }
  // MxK, NxK
  else if (!cargo_.transpose_a && cargo_.transpose_b) {
    MLLM_RT_ASSERT_EQ(shape_a[size_a - 1], shape_b[size_b - 1]);
    shape_c.push_back(shape_a[size_a - 2]);
    shape_c.push_back(shape_b[size_b - 2]);
  }
  // not supported
  else {
    NYI("MatMulOp::reshape with transpose_a={} and transpose_b={} is not supported yet",
        cargo_.transpose_a, cargo_.transpose_b);
  }

  // wrap to tensor
  auto o = Tensor::empty(shape_c, inputs[0].dtype(), inputs[0].device());
  outputs.emplace_back(o);
}

void MatMulOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  outputs[0].alloc();
}

}  // namespace mllm
