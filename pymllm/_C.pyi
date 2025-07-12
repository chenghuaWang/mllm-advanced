from __future__ import annotations
import typing
__all__ = ['BackendBase', 'DeviceTypes', 'HierarchyBase', 'HierarchyTypes', 'MemManager', 'MllmEngineCtx', 'OpTypes', 'X86Backend', 'create_x86_backend', 'get_engine_ctx']
class BackendBase:
    pass
class DeviceTypes:
    """
    Members:
    
      CPU
    
      CUDA
    
      OpenCL
    """
    CPU: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.CPU: 1>
    CUDA: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.CUDA: 2>
    OpenCL: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.OpenCL: 3>
    __members__: typing.ClassVar[dict[str, DeviceTypes]]  # value = {'CPU': <DeviceTypes.CPU: 1>, 'CUDA': <DeviceTypes.CUDA: 2>, 'OpenCL': <DeviceTypes.OpenCL: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class HierarchyBase:
    def absolute_name(self) -> str:
        ...
    def depth(self) -> int:
        ...
    def depth_decrease(self) -> None:
        ...
    def depth_increase(self) -> None:
        ...
    def is_compiled_as_obj(self) -> bool:
        ...
    def name(self) -> str:
        ...
    def set_absolute_name(self, arg0: str) -> None:
        ...
    def set_compiled_as_obj(self, arg0: bool) -> None:
        ...
    def set_depth(self, arg0: str) -> None:
        ...
    def set_name(self, arg0: str) -> None:
        ...
class HierarchyTypes:
    """
    Members:
    
      Module
    
      Layer
    """
    Layer: typing.ClassVar[HierarchyTypes]  # value = <HierarchyTypes.Layer: 1>
    Module: typing.ClassVar[HierarchyTypes]  # value = <HierarchyTypes.Module: 0>
    __members__: typing.ClassVar[dict[str, HierarchyTypes]]  # value = {'Module': <HierarchyTypes.Module: 0>, 'Layer': <HierarchyTypes.Layer: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MemManager:
    def init_buddy_ctx(self, arg0: DeviceTypes) -> None:
        ...
    def init_oc(self, arg0: DeviceTypes) -> None:
        ...
class MllmEngineCtx:
    def mem(self) -> MemManager:
        ...
    def register_backend(self, arg0: BackendBase) -> None:
        ...
    def shutdown(self) -> None:
        ...
class OpTypes:
    """
    Members:
    
      OpType_Start
    
      Fill
    
      Add
    
      Sub
    
      Mul
    
      Div
    
      MatMul
    
      LLMEmbeddingToken
    
      Linear
    
      RoPE
    
      Softmax
    
      Transpose
    
      RMSNorm
    
      SiLU
    
      KVCache
    
      CausalMask
    
      CastType
    
      D2H
    
      H2D
    
      Split
    
      View
    
      FlashAttention_2
    
      OpType_End
    """
    Add: typing.ClassVar[OpTypes]  # value = <OpTypes.Add: 2>
    CastType: typing.ClassVar[OpTypes]  # value = <OpTypes.CastType: 16>
    CausalMask: typing.ClassVar[OpTypes]  # value = <OpTypes.CausalMask: 15>
    D2H: typing.ClassVar[OpTypes]  # value = <OpTypes.D2H: 17>
    Div: typing.ClassVar[OpTypes]  # value = <OpTypes.Div: 5>
    Fill: typing.ClassVar[OpTypes]  # value = <OpTypes.Fill: 1>
    FlashAttention_2: typing.ClassVar[OpTypes]  # value = <OpTypes.FlashAttention_2: 21>
    H2D: typing.ClassVar[OpTypes]  # value = <OpTypes.H2D: 18>
    KVCache: typing.ClassVar[OpTypes]  # value = <OpTypes.KVCache: 14>
    LLMEmbeddingToken: typing.ClassVar[OpTypes]  # value = <OpTypes.LLMEmbeddingToken: 7>
    Linear: typing.ClassVar[OpTypes]  # value = <OpTypes.Linear: 8>
    MatMul: typing.ClassVar[OpTypes]  # value = <OpTypes.MatMul: 6>
    Mul: typing.ClassVar[OpTypes]  # value = <OpTypes.Mul: 4>
    OpType_End: typing.ClassVar[OpTypes]  # value = <OpTypes.OpType_End: 32>
    OpType_Start: typing.ClassVar[OpTypes]  # value = <OpTypes.OpType_Start: 0>
    RMSNorm: typing.ClassVar[OpTypes]  # value = <OpTypes.RMSNorm: 12>
    RoPE: typing.ClassVar[OpTypes]  # value = <OpTypes.RoPE: 9>
    SiLU: typing.ClassVar[OpTypes]  # value = <OpTypes.SiLU: 13>
    Softmax: typing.ClassVar[OpTypes]  # value = <OpTypes.Softmax: 10>
    Split: typing.ClassVar[OpTypes]  # value = <OpTypes.Split: 19>
    Sub: typing.ClassVar[OpTypes]  # value = <OpTypes.Sub: 3>
    Transpose: typing.ClassVar[OpTypes]  # value = <OpTypes.Transpose: 11>
    View: typing.ClassVar[OpTypes]  # value = <OpTypes.View: 20>
    __members__: typing.ClassVar[dict[str, OpTypes]]  # value = {'OpType_Start': <OpTypes.OpType_Start: 0>, 'Fill': <OpTypes.Fill: 1>, 'Add': <OpTypes.Add: 2>, 'Sub': <OpTypes.Sub: 3>, 'Mul': <OpTypes.Mul: 4>, 'Div': <OpTypes.Div: 5>, 'MatMul': <OpTypes.MatMul: 6>, 'LLMEmbeddingToken': <OpTypes.LLMEmbeddingToken: 7>, 'Linear': <OpTypes.Linear: 8>, 'RoPE': <OpTypes.RoPE: 9>, 'Softmax': <OpTypes.Softmax: 10>, 'Transpose': <OpTypes.Transpose: 11>, 'RMSNorm': <OpTypes.RMSNorm: 12>, 'SiLU': <OpTypes.SiLU: 13>, 'KVCache': <OpTypes.KVCache: 14>, 'CausalMask': <OpTypes.CausalMask: 15>, 'CastType': <OpTypes.CastType: 16>, 'D2H': <OpTypes.D2H: 17>, 'H2D': <OpTypes.H2D: 18>, 'Split': <OpTypes.Split: 19>, 'View': <OpTypes.View: 20>, 'FlashAttention_2': <OpTypes.FlashAttention_2: 21>, 'OpType_End': <OpTypes.OpType_End: 32>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class X86Backend(BackendBase):
    pass
def create_x86_backend() -> X86Backend:
    ...
def get_engine_ctx() -> MllmEngineCtx:
    """
    get the singleton instance of the engine context
    """
