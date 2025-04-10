import tilelang
import tilelang.language as T
from .tlop import TLOp


# ref:
# https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_gqa_fwd_bshd.py
def fa2_gqa_bshd(batch, seq_len, heads, dim, is_causal, groups=1, tune=False):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    inputs_dtype = "bfloat16"
    accumulate_dtype = "float32"

    def kernel_impl(block_M, block_N, num_stages, threads):
        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, inputs_dtype),  # type: ignore
            Q_shared: T.SharedBuffer([block_M, dim], inputs_dtype),  # type: ignore
            K_shared: T.SharedBuffer([block_N, dim], inputs_dtype),  # type: ignore
            acc_s: T.FragmentBuffer([block_M, block_N], accumulate_dtype),  # type: ignore
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)
            T.gemm(
                Q_shared,
                K_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, inputs_dtype),  # type: ignore
            V_shared: T.SharedBuffer([block_M, dim], inputs_dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_M, block_N], inputs_dtype),  # type: ignore
            acc_o: T.FragmentBuffer([block_M, dim], accumulate_dtype),  # type: ignore
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accumulate_dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_M, block_N], inputs_dtype),  # type: ignore
            scores_max: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
            scores_max_prev: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
            scores_scale: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
            scores_sum: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
            logsum: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accumulate_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accumulate_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(
                    scores_max_prev[i] * scale - scores_max[i] * scale
                )
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accumulate_dtype),  # type: ignore
            scores_scale: T.FragmentBuffer([block_M], accumulate_dtype),  # type: ignore
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
            Q: T.Buffer(q_shape, inputs_dtype),  # type: ignore
            K: T.Buffer(kv_shape, inputs_dtype),  # type: ignore
            V: T.Buffer(kv_shape, inputs_dtype),  # type: ignore
            Output: T.Buffer(q_shape, inputs_dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(seq_len, block_M), heads, batch, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], inputs_dtype)
                K_shared = T.alloc_shared([block_N, dim], inputs_dtype)
                V_shared = T.alloc_shared([block_N, dim], inputs_dtype)
                O_shared = T.alloc_shared([block_M, dim], inputs_dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accumulate_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], inputs_dtype)
                acc_o = T.alloc_fragment([block_M, dim], accumulate_dtype)
                scores_max = T.alloc_fragment([block_M], accumulate_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accumulate_dtype)
                scores_scale = T.alloc_fragment([block_M], accumulate_dtype)
                scores_sum = T.alloc_fragment([block_M], accumulate_dtype)
                logsum = T.alloc_fragment([block_M], accumulate_dtype)

                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accumulate_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_len, block_N),
                        T.ceildiv((bx + 1) * block_M, block_N),
                    )
                    if is_causal
                    else T.ceildiv(seq_len, block_N)
                )

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(
                        acc_s,
                        acc_s_cast,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                    )
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        return main

    if tune:

        raise NotImplementedError("Tuning is not implemented yet.")
    else:

        def kernel(block_M, block_N, num_stages, threads):
            return kernel_impl(block_M, block_N, num_stages, threads)

        return kernel
