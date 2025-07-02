/**
 * @file LinearOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "mllm/Core/AOps/LinearOp.hpp"
#include "mllm/Backends/QNN/Ops/QnnBaseOp.hpp"

// MLLM Documents:
//
// This linear op in mllm's qnn backend is optimized for:
//
// activation shape: [batch, in_channel]
// weight shape: [in_channel, out_channel]
// bias shape: [out_channel]

//
// QNN Documents:
//
// The FullyConnected operation connects the all input elements with each output element through
// weights and biases. The weights tensor has shape [m, n] where n is the number of input elements
// and m is the units of weights, output and optional biases. The input activation must be
// reshapable to [batch, n] (see Reshape operation definition) and the operation computes
// mathematically:
//
// outputVector = ( inputAsVector * weightsMatrix ) + biasesVector
//
// Inputs
// in[0]
// input activation
// Mandatory: true
// Data type: backend specific
// Shape: [n] or Rank >= 2 reshapable to [batch, n]
// Dynamic Shape: All dimensions can be dynamic.
//
// Constraints:
// Shape: Rank > 0
// Shape: Rank >= 2 must be reshapable to [batch, n]
//
// in[1]
// weights
// Mandatory: true
// Data type: backend specific
// Shape: [m, n]
// Dynamic Shape: All dimensions can be dynamic.
//
// Constraints:
// Dynamic Shape: if rank(in[0]) = 1, then shape(in[0])[n] and shape(in[1])[n] must be both dynamic
// or both static. Otherwise if rank(in[0]) > 1, shape(in[1])[n] must be static.

// in[2]
// biases
// Mandatory: false
// Data type: backend specific
// Shape: [m]
// Dynamic Shape: All dimensions can be dynamic.
// Default: [0]
// Constraints:
// Dynamic Shape: shape(in[1])[m] and shape(in[2])[m] must be both dynamic or both static.
//
// Parameters
// keep_dims
// If true, the rank of in[0] and out[0] will remain the same, and all but the last dimension will
// be equal in shape. For dimensions to be preserved, the product of the batch dimensions of in[0]
// (all but the last dimension) must be equal to batch, defined by in[0] above. This is because:
// total # of outputs = (total # of batches) * m
// Since the total # of outputs and m are the same regardless of keep_dims, the total # of batches
// must remain the same as well. Mandatory: false Data type: QNN_DATATYPE_BOOL_8 Shape: scalar
// Default: 0
//
// Outputs
// out[0]
// output activation
// Mandatory: true
// Data type: backend specific
// Shape: If the rank of in[0] is 1: [m]. If the rank of in[0] is > 1: [batch, m], unless keep_dims
// is true, then […, m] where … is all but the last dimension of in[0] Dynamic Shape: All dimensions
// can be dynamic. Constraints: Datatype: Same datatype as in[0] Dynamic Shape: If rank(in[0]) > 1
// and keep_dims is set to false and any dimension for in[0] is dynamic then shape(out[0])[batch]
// must be dynamic. Otherwise, if rank(in[0]) > 1 and keep_dims is set to true, for each dimension
// with the exemption of the last dimension, if shape(in[0])[i] is dynamic then shape(out[0])[i]
// must be dynamic. Dynamic Shape: If shape(in[1])[m] is dynamic then shape(out[0])[m] must be
// dynamic.

namespace mllm::qnn {

class QnnLinearOpPattern final : public QnnBaseOpPattern {
 public:
  bool match(const ir::op_ptr_t& op) override;

  bool addNode(QnnIRGraph& graph, const ir::op_ptr_t& op,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::self_ptr_t>& outputs) override;

  static std::pair<OpType, std::shared_ptr<QnnBaseOpPattern>> create();
};

class QnnLinearOp final : public LinearOp {
 public:
  explicit QnnLinearOp(const LinearOpCargo& cargo);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  // NOTE: There is no need to override `reshape` function.

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class QnnLinearOpFactory : public TypedOpFactory<OpType::kLinear, LinearOpCargo> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const LinearOpCargo& cargo) override {
    return std::make_shared<QnnLinearOp>(cargo);
  }
};

}  // namespace mllm::qnn
