import tilelang
import tilelang.language as T


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
