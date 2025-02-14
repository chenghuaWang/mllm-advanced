// Auto generated: 2025-02-14 18:15:23
// do not modify this file
#pragma once
namespace mllm::ir {

// traits
#ifdef RTTI_NODE_IMPL
template<typename T>
struct NodeRTTIClassOfImpl {
  static inline bool classof(Node* v) { return false; }
};
#endif  //! RTTI_Node_IMPL

#define RTTI_RK_OP_IMPL(v) return (v)->getKind() >= RK_Op && (v)->getKind() <= RK_Op_Last

#define RTTI_RK_OP_LINALGIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp && (v)->getKind() <= RK_Op_LinalgIROp_Last

#define RTTI_RK_OP_LINALGIROP_CUSTOMKERNELOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_CustomKernelOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_CustomKernelOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_Last

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EADDOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EAddOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EAddOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_ESUBOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_ESubOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_ESubOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EMULOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EMulOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EMulOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EDIVOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EDivOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EDivOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EBROADCASTADDOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastAddOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastAddOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EBROADCASTSUBOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastSubOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastSubOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EBROADCASTMULOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastMulOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastMulOp

#define RTTI_RK_OP_LINALGIROP_ELEMENTWISEOP_EBROADCASTDIVOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastDivOp \
         && (v)->getKind() <= RK_Op_LinalgIROp_ElementwiseOp_EBroadcastDivOp

#define RTTI_RK_OP_LINALGIROP_MATMULOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_MatMulOp && (v)->getKind() <= RK_Op_LinalgIROp_MatMulOp

#define RTTI_RK_OP_LINALGIROP_LINEAROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_LinearOp && (v)->getKind() <= RK_Op_LinalgIROp_LinearOp

#define RTTI_RK_OP_LINALGIROP_SDPAOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SDPAOp && (v)->getKind() <= RK_Op_LinalgIROp_SDPAOp

#define RTTI_RK_OP_LINALGIROP_ROPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RoPEOp && (v)->getKind() <= RK_Op_LinalgIROp_RoPEOp

#define RTTI_RK_OP_LINALGIROP_ROPE2DOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RoPE2dOp && (v)->getKind() <= RK_Op_LinalgIROp_RoPE2dOp

#define RTTI_RK_OP_LINALGIROP_CONV2DOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_Conv2dOp && (v)->getKind() <= RK_Op_LinalgIROp_Conv2dOp

#define RTTI_RK_OP_GRAPHIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_GraphIROp && (v)->getKind() <= RK_Op_GraphIROp_Last

#define RTTI_RK_OP_GRAPHIROP_SUBGRAPHOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_GraphIROp_SubGraphOp \
         && (v)->getKind() <= RK_Op_GraphIROp_SubGraphOp

#define RTTI_RK_OP_GRAPHIROP_CALLGRAPHOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_GraphIROp_CallGraphOp \
         && (v)->getKind() <= RK_Op_GraphIROp_CallGraphOp

#define RTTI_RK_OP_TENSORIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp && (v)->getKind() <= RK_Op_TensorIROp_Last

#define RTTI_RK_OP_TENSORIROP_ALLOCOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp_AllocOp && (v)->getKind() <= RK_Op_TensorIROp_AllocOp

#define RTTI_RK_OP_TENSORIROP_ALLOCGLOBALOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_TensorIROp_AllocGlobalOp \
         && (v)->getKind() <= RK_Op_TensorIROp_AllocGlobalOp

#define RTTI_RK_OP_TENSORIROP_FREEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp_FreeOp && (v)->getKind() <= RK_Op_TensorIROp_FreeOp

#define RTTI_RK_OP_BUILTINIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_BuiltinIROp && (v)->getKind() <= RK_Op_BuiltinIROp_Last

#define RTTI_RK_OP_BUILTINIROP_MODULEOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_BuiltinIROp_ModuleOp \
         && (v)->getKind() <= RK_Op_BuiltinIROp_ModuleOp

#define RTTI_RK_OP_CONTROLFLOWIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ControlFlowIROp && (v)->getKind() <= RK_Op_ControlFlowIROp_Last

#define RTTI_RK_OP_CONTROLFLOWIROP_RETURNOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ControlFlowIROp_ReturnOp \
         && (v)->getKind() <= RK_Op_ControlFlowIROp_ReturnOp

#define RTTI_RK_OP_PROGRAMIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp && (v)->getKind() <= RK_Op_ProgramIROp_Last

#define RTTI_RK_OP_PROGRAMIROP_OBJECTFRAGMENTOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_ObjectFragmentOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_ObjectFragmentOp

#define RTTI_RK_OP_PROGRAMIROP_DATAFRAGMENTOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_DataFragmentOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_DataFragmentOp

#define RTTI_RK_OP_PROGRAMIROP_GLOBALDATAFRAGMENTOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_GlobalDataFragmentOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_GlobalDataFragmentOp

#define RTTI_RK_OP_PROGRAMIROP_INSTRUCTIONOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_InstructionOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_InstructionOp

#define RTTI_RK_OP_PROGRAMIROP_GLOBALDATAITEMOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_GlobalDataItemOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_GlobalDataItemOp

#define RTTI_RK_OP_PROGRAMIROP_DATAITEMOP_IMPL(v)       \
  return (v)->getKind() >= RK_Op_ProgramIROp_DataItemOp \
         && (v)->getKind() <= RK_Op_ProgramIROp_DataItemOp

#define RTTI_RK_VAL_IMPL(v) return (v)->getKind() >= RK_Val && (v)->getKind() <= RK_Val_Last

#define RTTI_RK_VAL_LINALGIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_LinalgIRVal && (v)->getKind() <= RK_Val_LinalgIRVal

#define RTTI_RK_VAL_GRAPHIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_GraphIRVal && (v)->getKind() <= RK_Val_GraphIRVal

#define RTTI_RK_VAL_TENSORIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_TensorIRVal && (v)->getKind() <= RK_Val_TensorIRVal_Last

#define RTTI_RK_VAL_TENSORIRVAL_TENSORVAL_IMPL(v)       \
  return (v)->getKind() >= RK_Val_TensorIRVal_TensorVal \
         && (v)->getKind() <= RK_Val_TensorIRVal_TensorVal

#define RTTI_RK_VAL_BUILTINIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_BuiltinIRVal && (v)->getKind() <= RK_Val_BuiltinIRVal

#define RTTI_RK_VAL_CONTROLFLOWIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_ControlFlowIRVal && (v)->getKind() <= RK_Val_ControlFlowIRVal

#define RTTI_RK_VAL_PROGRAMIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_ProgramIRVal && (v)->getKind() <= RK_Val_ProgramIRVal

#define RTTI_RK_ATTR_IMPL(v) return (v)->getKind() >= RK_Attr && (v)->getKind() <= RK_Attr_Last

#define RTTI_RK_ATTR_LINALGIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_LinalgIRAttr && (v)->getKind() <= RK_Attr_LinalgIRAttr

#define RTTI_RK_ATTR_GRAPHIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_GraphIRAttr && (v)->getKind() <= RK_Attr_GraphIRAttr

#define RTTI_RK_ATTR_TENSORIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_TensorIRAttr && (v)->getKind() <= RK_Attr_TensorIRAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_Last

#define RTTI_RK_ATTR_BUILTINIRATTR_INTATTR_IMPL(v)       \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_IntAttr \
         && (v)->getKind() <= RK_Attr_BuiltinIRAttr_IntAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_FPATTR_IMPL(v)       \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_FPAttr \
         && (v)->getKind() <= RK_Attr_BuiltinIRAttr_FPAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_STRATTR_IMPL(v)       \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_StrAttr \
         && (v)->getKind() <= RK_Attr_BuiltinIRAttr_StrAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_SYMBOLATTR_IMPL(v)       \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_SymbolAttr \
         && (v)->getKind() <= RK_Attr_BuiltinIRAttr_SymbolAttr

#define RTTI_RK_ATTR_CONTROLFLOWIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_ControlFlowIRAttr && (v)->getKind() <= RK_Attr_ControlFlowIRAttr

#define RTTI_RK_ATTR_PROGRAMIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_ProgramIRAttr && (v)->getKind() <= RK_Attr_ProgramIRAttr

}  // namespace mllm::ir
