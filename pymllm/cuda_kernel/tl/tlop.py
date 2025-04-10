from abc import ABC, abstractmethod
from typing import Dict


class TLOp(ABC):
    def __init__(self, op_type, op_name: str, kernel_handle):
        self.op_type = op_type
        self.op_name: str = op_name
        self.kernel_handle = kernel_handle
        self.compiled_obj: Dict[str, object] = {}

    @abstractmethod
    def selector(self, *argv, **kwargs):
        raise NotImplementedError(
            f"TLOp: {self.op_name}'s selector func is not implemented!"
        )

    @abstractmethod
    def compile(self, *argv, **kwargs):
        raise NotImplementedError(
            f"TLOp: {self.op_name}'s compile func is not implemented!"
        )

    @abstractmethod
    def forward(self, *argv, **kwargs):
        raise NotImplementedError(
            f"TLOp: {self.op_name}'s compile func is not implemented!"
        )
