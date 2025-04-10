import tilelang
import tilelang.language as T
from .tlop import TLOp


def elementwise_add(
    M,
    N,
    block_M,
    block_N,
    in_dtype,
    out_dtype,
    threads,
):

    @T.prim_func
    def main(
        A: T.Buffer((M, N), in_dtype),  # type: ignore
        B: T.Buffer((M, N), in_dtype),  # type: ignore
        C: T.Buffer((M, N), out_dtype),  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return main


class ElewiseAddTLOp(TLOp):
    def __init__(
        self, op_type, op_name="element_wise_add", kernel_handle=elementwise_add
    ):
        super().__init__(op_type, op_name, kernel_handle)
        self.kernel_cfg = {
            # general purpose. fp32
            "m_n_bm16_bn16_t256_fp32_fp32": {
                "M": T.symbolic("M"),
                "N": T.symbolic("N"),
                "block_M": 128,
                "block_N": 256,
                "in_dtype": "float32",
                "out_dtype": "float32",
                "threads": 256,
            },
            # general purpose. fp16
            "m_n_bm16_bn16_t256_fp16_fp16": {
                "M": T.symbolic("M"),
                "N": T.symbolic("N"),
                "block_M": 128,
                "block_N": 256,
                "in_dtype": "float16",
                "out_dtype": "float16",
                "threads": 256,
            },
        }

    def selector(self, A, B):
        # TODO use dtype and size to judge.
        return "m_n_bm16_bn16_t256_fp32_fp32"

    def compile(self, *argv, **kwargs):
        for key, cfg in self.kernel_cfg.items():
            program = self.kernel_handle(**cfg)
            kernel = tilelang.compile(
                program, out_idx=-1, target="cuda", execution_backend="cython"
            )
            self.compiled_obj[key] = kernel

    def forward(self, A, B):
        selected = self.selector(A, B)
        return self.compiled_obj[selected](A, B)
