/**
 * @file Op.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-03-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/IR/Linalg/Op.hpp"
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm::ir::linalg {

LinalgIROp::LinalgIROp() : Op(RK_Op_LinalgIROp) {}

LinalgIROp::LinalgIROp(const NodeKind& kind) : Op(kind) {}

LINALG_AOPS_DECL(OpType::kFill, FillOp)
LINALG_AOPS_DECL(OpType::kAdd, AddOp);
LINALG_AOPS_DECL(OpType::kSub, SubOp);
LINALG_AOPS_DECL(OpType::kMul, MulOp);
LINALG_AOPS_DECL(OpType::kDiv, DivOp);

LINALG_AOPS_DECL(OpType::kMatMul, MatMulOp);

LINALG_AOPS_DECL(OpType::kLLMEmbeddingToken, LLMEmbeddingTokenOp);
LINALG_AOPS_DECL(OpType::kLinear, LinearOp);
LINALG_AOPS_DECL(OpType::kRoPE, RoPEOp);
LINALG_AOPS_DECL(OpType::kKVCache, KVCacheOp);
LINALG_AOPS_DECL(OpType::kCausalMask, CausalMaskOp);

LINALG_AOPS_DECL(OpType::kSoftmax, SoftmaxOp);
LINALG_AOPS_DECL(OpType::kTranspose, TransposeOp);
LINALG_AOPS_DECL(OpType::kRMSNorm, RMSNormOp);
LINALG_AOPS_DECL(OpType::kSiLU, SiLUOp);

LINALG_AOPS_DECL(OpType::kCastType, CastTypeOp);

LINALG_AOPS_DECL(OpType::kD2H, D2HOp);

LINALG_AOPS_DECL(OpType::kView, ViewOp);
LINALG_AOPS_DECL(OpType::kSplit, SplitOp);

LINALG_AOPS_DECL(OpType::kFlashAttention_2, FlashAttention2Op);
LINALG_AOPS_DECL(OpType::kRepeat, RepeatOp);
LINALG_AOPS_DECL(OpType::kPermute, PermuteOp);

LINALG_AOPS_DECL(OpType::kConv1D, Conv1DOp);
LINALG_AOPS_DECL(OpType::kConv2D, Conv2DOp);
LINALG_AOPS_DECL(OpType::kConv3D, Conv3DOp);

LINALG_AOPS_DECL(OpType::kGELU, GELUOp);
LINALG_AOPS_DECL(OpType::kLayerNorm, LayerNormOp);

LINALG_AOPS_DECL(OpType::kMultimodalRoPE, MultimodalRoPEOp);
LINALG_AOPS_DECL(OpType::kVisionRoPE, VisionRoPEOp);

LINALG_AOPS_DECL(OpType::kQuickGELU, QuickGELUOp);

LINALG_AOPS_DECL(OpType::kCopy, CopyOp);
LINALG_AOPS_DECL(OpType::kClone, CloneOp);

}  // namespace mllm::ir::linalg
