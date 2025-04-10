import torch
from pymllm.cuda_kernel.tl.elewise import ElewiseAddTLOp


def ref_program(x, y):
    return x + y


def test_elewise_add():
    op = ElewiseAddTLOp(None)
    op.compile()
    A = torch.ones(1024, 1024, dtype=torch.float32).cuda()
    B = torch.ones(1024, 1024, dtype=torch.float32).cuda()
    C = op.forward(A, B)
    assert torch.allclose(C, ref_program(A, B))
