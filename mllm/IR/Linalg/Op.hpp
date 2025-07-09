/**
 * @file Op.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include "mllm/Core/AOps/BaseOp.hpp"
#include "mllm/Core/DeviceTypes.hpp"
#include "mllm/IR/Node.hpp"
#include "mllm/IR/Tensor/Value.hpp"

namespace mllm {
class BaseOp;
class AddOp;
class SubOp;
class MulOp;
class DivOp;
class FillOp;
class MatMulOp;
class LLMEmbeddingTokenOp;
class LinearOp;
class RoPEOp;
class KVCacheOp;
class SoftmaxOp;
class TransposeOp;
class RMSNormOp;
class SiLUOp;
class CausalMaskOp;
class CastTypeOp;
class D2HOp;
class H2DOp;
class ViewOp;
class SplitOp;
class FlashAttention2Op;
class RepeatOp;
class PermuteOp;
class Conv1DOp;
class Conv2DOp;
class Conv3DOp;
class GELUOp;
class LayerNormOp;
class MultimodalRoPEOp;
class VisionRoPEOp;
}  // namespace mllm

#define LINALG_AOPS_DEFINE(class_name, rtti_name)                                  \
  class class_name final : public LinalgIROp {                                     \
   public:                                                                         \
    DEFINE_SPECIFIC_IR_CLASS(class_name);                                          \
    ~class_name() override;                                                        \
    class_name();                                                                  \
    explicit class_name(const std::shared_ptr<BaseOp>& aop);                       \
    ::mllm::class_name* getOp() { return (::mllm::class_name*)(op_.get()); }       \
    static inline bool classof(const Node* node) {                                 \
      RTTI_RK_OP_LINALGIROP_##rtti_name##_IMPL(node);                              \
    }                                                                              \
    static std::shared_ptr<::mllm::ir::linalg::class_name> build(                  \
        IRContext* ctx, const std::shared_ptr<BaseOp>& aop,                        \
        const std::vector<std::shared_ptr<::mllm::ir::tensor::TensorValue>>& ins,  \
        const std::vector<std::shared_ptr<::mllm::ir::tensor::TensorValue>>& ous); \
    void dump(IRPrinter& p) override;                                              \
  }

#define LINALG_AOPS_DECL(op_type, class_name)                                     \
  class_name::~class_name() = default;                                            \
  class_name::class_name(const std::shared_ptr<BaseOp>& aop)                      \
      : LinalgIROp(RK_Op_LinalgIROp_##class_name) {                               \
    setAOp(op_type, aop);                                                         \
  }                                                                               \
  std::shared_ptr<::mllm::ir::linalg::class_name> class_name::build(              \
      IRContext* ctx, const std::shared_ptr<BaseOp>& aop,                         \
      const std::vector<std::shared_ptr<::mllm::ir::tensor::TensorValue>>& ins,   \
      const std::vector<std::shared_ptr<::mllm::ir::tensor::TensorValue>>& ous) { \
    auto op = std::make_shared<::mllm::ir::linalg::class_name>(aop);              \
    for (auto& i : ins) { (*i)-- > op; }                                          \
    for (auto& o : ous) { (*op)-- > o; }                                          \
    return op;                                                                    \
  }                                                                               \
  void class_name::dump(IRPrinter& p) {                                           \
    p.print("linalg.{}.{}", deviceTypes2Str(getDevice()), #class_name);           \
    Op::dump(p);                                                                  \
  }

namespace mllm::ir::linalg {
class LinalgIROp : public Op {
 public:
  DEFINE_SPECIFIC_IR_CLASS(LinalgIROp);

  ~LinalgIROp() override = default;
  LinalgIROp();
  explicit LinalgIROp(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_OP_LINALGIROP_IMPL(node); }

  inline void setAOp(OpType op_type, const std::shared_ptr<BaseOp>& op) {
    op_type_ = op_type;
    op_ = op;
  }

  inline OpType getAOpType() const { return op_type_; }

  inline BaseOp* getAOp() const { return op_.get(); }

 protected:
  OpType op_type_;
  std::shared_ptr<BaseOp> op_;
};

LINALG_AOPS_DEFINE(FillOp, FILLOP);
LINALG_AOPS_DEFINE(AddOp, ADDOP);
LINALG_AOPS_DEFINE(SubOp, SUBOP);
LINALG_AOPS_DEFINE(MulOp, MULOP);
LINALG_AOPS_DEFINE(DivOp, DIVOP);

LINALG_AOPS_DEFINE(MatMulOp, MATMULOP);

LINALG_AOPS_DEFINE(LLMEmbeddingTokenOp, LLMEMBEDDINGTOKENOP);
LINALG_AOPS_DEFINE(LinearOp, LINEAROP);
LINALG_AOPS_DEFINE(RoPEOp, ROPEOP);
LINALG_AOPS_DEFINE(KVCacheOp, KVCACHEOP);
LINALG_AOPS_DEFINE(CausalMaskOp, CAUSALMASKOP);

LINALG_AOPS_DEFINE(SoftmaxOp, SOFTMAXOP);
LINALG_AOPS_DEFINE(TransposeOp, TRANSPOSEOP);
LINALG_AOPS_DEFINE(RMSNormOp, RMSNORMOP);
LINALG_AOPS_DEFINE(SiLUOp, SILUOP);

LINALG_AOPS_DEFINE(CastTypeOp, CASTTYPEOP);

LINALG_AOPS_DEFINE(D2HOp, D2HOP);

LINALG_AOPS_DEFINE(ViewOp, VIEWOP);
LINALG_AOPS_DEFINE(SplitOp, SPLITOP);

LINALG_AOPS_DEFINE(FlashAttention2Op, FLASHATTENTION2OP);
LINALG_AOPS_DEFINE(RepeatOp, REPEATOP);
LINALG_AOPS_DEFINE(PermuteOp, PERMUTEOP);

LINALG_AOPS_DEFINE(Conv1DOp, CONV1DOP);
LINALG_AOPS_DEFINE(Conv2DOp, CONV2DOP);
LINALG_AOPS_DEFINE(Conv3DOp, CONV3DOP);

LINALG_AOPS_DEFINE(GELUOp, GELUOP);
LINALG_AOPS_DEFINE(LayerNormOp, LAYERNORMOP);

LINALG_AOPS_DEFINE(MultimodalRoPEOp, MULTIMODALROPEOP);
LINALG_AOPS_DEFINE(VisionRoPEOp, VISIONROPEOP);

}  // namespace mllm::ir::linalg