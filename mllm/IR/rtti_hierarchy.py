"""
File for gen_rtti_enum tool to auto generate Node Kinds.

1. Base Classes

Node -> Op, Value, Attribute

2. Op/Value/Attribute Hierarchy

Op |
   LinalgOp |
            Kernel
            others
   GraphOp  |
            SubGraph
            GraphCall

Same as Value and Attribute
"""

from typing import List
from pathlib import Path
from datetime import datetime


class Cls:
    def __init__(self, name: str):
        self.name: str = name
        self.childs: List = []

        # for enum kind to generate cpp code
        self.enum_name: str = name
        self.classof_s = None
        self.classof_e = None

    def derive(self, cls):
        self.childs.append(cls)
        return cls


class RTTIGenCPPDumper:
    def __init__(self, name_space: str):
        self.template: str = """\
// Auto generated: <head>
// do not modify this file
#pragma once

#include <cstdint>

namespace <namespace> {

<kind>

class <top_cls_name>;
template<typename T>
struct <top_cls_name>RTTIClassOfImpl {
  static inline bool classof(<top_cls_name>* v) { return false; }
};
}
        """
        now = datetime.now()
        date_time_format = now.strftime("%Y-%m-%d %H:%M:%S")
        self.template = self.template.replace("<head>", date_time_format)
        self.template = self.template.replace("<namespace>", name_space)

    def set_kind(self, kinds: str):
        self.template = self.template.replace("<kind>", kinds)

    def set_top_cls(self, cls: Cls):
        self.template = self.template.replace("<top_cls_name>", cls.name)


class RTTIGenImplDumper:
    def __init__(self, name_space):
        self.template: str = """\
// Auto generated: <head>
// do not modify this file
#pragma once
namespace <namespace> {

<content>
}
"""
        now = datetime.now()
        date_time_format = now.strftime("%Y-%m-%d %H:%M:%S")
        self.template = self.template.replace("<head>", date_time_format)
        self.template = self.template.replace("<namespace>", name_space)
        self.content: List[str] = []

        self.base_impl_template = """\
// traits
#ifdef RTTI_<top_cls_name_upper>_IMPL
template<typename T>
struct <top_cls_name>RTTIClassOfImpl {
  static inline bool classof(<top_cls_name>* v) { return false; }
};
#endif //! RTTI_<top_cls_name>_IMPL
"""
        self.other_impl_template = """\
#define RTTI_<class_name>_IMPL(v) \\
    return (v)->getKind() >= <cls_s> && (v)->getKind() <= <cls_e> \\
"""
        self.top_cls_name: str = ""

    def gen_base_impl(self, cls: Cls):
        self.top_cls_name = cls.name
        self.content.append(
            self.base_impl_template.replace("<top_cls_name>", cls.name).replace(
                "<top_cls_name_upper>", cls.name.upper()
            )
        )

    def gen_a_impl(self, cls: Cls):
        o = self.other_impl_template.replace("<top_cls_name>", self.top_cls_name)
        o = o.replace("<T>", cls.name)
        o = o.replace("<class_name>", cls.enum_name.upper())
        o = o.replace("<cls_s>", cls.classof_s)
        o = o.replace("<cls_e>", cls.classof_e)
        self.content.append(o)

    def gen_impl(self, cls: Cls):
        for sub_class in cls.childs:
            self.gen_a_impl(sub_class)
            if len(sub_class.childs) != 0:
                self.gen_impl(sub_class)

    def finalize(self, cls: Cls):
        self.gen_base_impl(cls)
        self.gen_impl(cls)
        self.template = self.template.replace("<content>", "\n".join(self.content))


def recursion_dump_cls(father_name: str, cls: Cls, rets: List[str]):
    if len(cls.childs) != 0:
        father_name = father_name + "_" + cls.name
        # insert start
        rets.append(father_name)
        cls.classof_s = rets[-1]
        cls.enum_name = rets[-1]

        # internals
        for sub_cls in cls.childs:
            recursion_dump_cls(father_name, sub_cls, rets)

        # insert last
        rets.append(father_name + "_Last")
        cls.classof_e = rets[-1]
    else:
        rets.append(father_name + "_" + cls.name)
        cls.enum_name = rets[-1]
        cls.classof_s = rets[-1]
        cls.classof_e = rets[-1]


def dump_cls_to_kinds(cls: Cls, ignore_top_level: bool = True) -> str:
    if not ignore_top_level:
        raise NotImplementedError

    base_template = """\
// <top_cls_name>Kind Enum Type for classof to use
// this enum is auto generated by rtti hierarchy generator.
// the enum class of LLVM style rtti.
//
// RTTI Kind (RK)
enum <top_cls_name>Kind : uint32_t {
<cls_kind_content>
};
    """
    base_template = base_template.replace("<top_cls_name>", cls.name)

    rets: List[str] = ["RK_None"]

    for sub_cls in cls.childs:
        recursion_dump_cls("RK", sub_cls, rets)

    rets.append("")
    base_template = base_template.replace("<cls_kind_content>", ",\n".join(rets))

    return base_template


def define_ir(name: str, op: Cls, val: Cls, attr: Cls):
    ret = {
        "Op": Cls(f"{name}Op"),
        "Value": Cls(f"{name}Val"),
        "Attribute": Cls(f"{name}Attr"),
    }
    op.derive(ret["Op"])
    val.derive(ret["Value"])
    attr.derive(ret["Attribute"])
    return ret


def define_builtin_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("ModuleOp"))

    # value

    # attr
    attr.derive(Cls("IntAttr"))
    attr.derive(Cls("FPAttr"))
    attr.derive(Cls("StrAttr"))
    attr.derive(Cls("SymbolAttr"))
    attr.derive(Cls("BoolAttr"))


def define_lianlg_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("CustomKernelOp"))
    op.derive(Cls("FillOp"))
    op.derive(Cls("AddOp"))
    op.derive(Cls("SubOp"))
    op.derive(Cls("MulOp"))
    op.derive(Cls("DivOp"))
    op.derive(Cls("MatMulOp"))
    op.derive(Cls("LLMEmbeddingTokenOp"))
    op.derive(Cls("LinearOp"))
    op.derive(Cls("RoPEOp"))
    op.derive(Cls("SoftmaxOp"))
    op.derive(Cls("TransposeOp"))
    op.derive(Cls("RMSNormOp"))
    op.derive(Cls("SiLUOp"))
    op.derive(Cls("KVCacheOp"))
    op.derive(Cls("CausalMaskOp"))
    op.derive(Cls("CastTypeOp"))
    op.derive(Cls("D2HOp"))
    op.derive(Cls("H2DOp"))
    op.derive(Cls("ViewOp"))
    op.derive(Cls("SplitOp"))
    op.derive(Cls("FlashAttention2Op"))
    op.derive(Cls("RepeatOp"))
    op.derive(Cls("PermuteOp"))
    op.derive(Cls("Conv1DOp"))
    op.derive(Cls("Conv2DOp"))
    op.derive(Cls("Conv3DOp"))
    op.derive(Cls("GELUOp"))
    op.derive(Cls("LayerNormOp"))
    op.derive(Cls("MultimodalRoPEOp"))
    op.derive(Cls("VisionRoPEOp"))
    op.derive(Cls("QuickGELUOp"))
    op.derive(Cls("CopyOp"))
    op.derive(Cls("CloneOp"))
    op.derive(Cls("NegOp"))
    op.derive(Cls("ConcatOp"))

    # value

    # attr


def define_graph_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("SubGraphOp"))
    op.derive(Cls("CallGraphOp"))

    # value

    # attr


def define_tensor_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("AllocOp"))
    op.derive(Cls("AllocGlobalOp"))
    op.derive(Cls("FreeOp"))

    # value
    val.derive(Cls("TensorVal"))

    # attr


def define_control_flow_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("ReturnOp"))


def define_program_ir(ir: dict):
    op: Cls = ir["Op"]
    val: Cls = ir["Value"]
    attr: Cls = ir["Attribute"]

    # op
    op.derive(Cls("ObjectFragmentOp"))
    op.derive(Cls("DataFragmentOp"))
    op.derive(Cls("GlobalDataFragmentOp"))
    op.derive(Cls("InstructionOp"))
    op.derive(Cls("GlobalDataItemOp"))
    op.derive(Cls("DataItemOp"))


if __name__ == "__main__":
    Node = Cls("Node")
    # Node Level
    Op = Node.derive(Cls("Op"))
    Val = Node.derive(Cls("Val"))
    Attr = Node.derive(Cls("Attr"))

    # Op Level
    linalg_ir = define_ir("LinalgIR", Op, Val, Attr)
    graph_ir = define_ir("GraphIR", Op, Val, Attr)
    tensor_ir = define_ir("TensorIR", Op, Val, Attr)
    builtin_ir = define_ir("BuiltinIR", Op, Val, Attr)
    cf_ir = define_ir("ControlFlowIR", Op, Val, Attr)
    program_ir = define_ir("ProgramIR", Op, Val, Attr)

    # define IR
    define_builtin_ir(builtin_ir)
    define_lianlg_ir(linalg_ir)
    define_graph_ir(graph_ir)
    define_tensor_ir(tensor_ir)
    define_control_flow_ir(cf_ir)
    define_program_ir(program_ir)

    d = RTTIGenCPPDumper("mllm::ir")
    d.set_kind(dump_cls_to_kinds(Node))
    d.set_top_cls(Node)

    impl_d = RTTIGenImplDumper("mllm::ir")
    impl_d.finalize(Node)

    with open(
        Path(__file__).parent / "GeneratedRTTIKind.hpp", "w", encoding="utf-8"
    ) as f:
        f.write(d.template)

    with open(
        Path(__file__).parent / "NodeRTTIClassOfImpl.hpp", "w", encoding="utf-8"
    ) as f:
        f.write(impl_d.template)
