from __future__ import annotations
import typing
__all__ = ['BackendBase', 'DeviceTypes', 'MemManager', 'MllmEngineCtx', 'X86Backend', 'create_x86_backend', 'get_engine_ctx']
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
class X86Backend(BackendBase):
    pass
def create_x86_backend() -> X86Backend:
    ...
def get_engine_ctx() -> MllmEngineCtx:
    """
    get the singleton instance of the engine context
    """
