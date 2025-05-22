from .tlop import TLOp
from ..base_op_factory import BaseDSLOpFactory

from .elewise import ElewiseAddTLOp
from .fa2 import FA2GqaBshdTLOp


class TLOpFactory(BaseDSLOpFactory):
    def __init__(self):
        super().__init__()
        self._ops = [
            ElewiseAddTLOp,
            FA2GqaBshdTLOp,
        ]
        for _op in self._ops:
            instanced = _op()
            self._register_op(instanced.op_name, instanced)

    def _register_op(self, op_type: str, op: TLOp):
        self.ops.update({op_type: op})

    def compile_all(self):
        for op_type, op in self.ops.item():
            op.compile()
