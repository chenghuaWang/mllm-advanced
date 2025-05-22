from abc import ABC
from typing import Dict
from .._C import OpTypes


class BaseDSLOpFactory(ABC):
    def __init__(self):
        # string : Op
        self.ops: Dict = {}

    def __str__(self):
        ret = []
        ret.append(self.__class__.__name__)
        for key, op in self.ops.items():
            ret.append(f"Type:{op.op_type}, Sp: {op.op_name}")
        return "\n".join(ret)

    def get(self, op_type: OpTypes, op_name: str):
        op = self.ops.get(op_name, None)
        assert op.op_type == op_type, print(f"{op.op_type} mismatch {op_type}")
        return op
