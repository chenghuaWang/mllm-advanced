from .tlop import TLOp
from typing import Dict


class TLOpFactory:
    def __init__(self):
        self.ops: Dict[str, TLOp] = {}

    def _register_op(self, op_name: str, op: TLOp):
        self.ops.update({op_name: op})

    def compile_all(self):
        pass
