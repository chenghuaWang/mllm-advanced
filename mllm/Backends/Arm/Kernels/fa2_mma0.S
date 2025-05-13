// ARM AArch64. ARM V8.2A with armv8.2-a+fp16+fp16fml+dotprod
// Optimized kernel for Flash Attention 2's MMA0
//
// MMA0 Kernel:
// Br=4, Cr=4, K is dividable by 8
// Q @ K^T.
//
// Inputs: Q_block: Br x K
// Inputs: K_block: Cr x k
// Output: Br x Cr

#include "mllm/Utils/AssemblyHelper.hpp"

MLLM_CPU_ASM_EXPORT(mllm_fa2_mma0_bshd_fp16_br4_bc4)
MLLM_CPU_ASM_FUNCTION(mllm_fa2_mma0_bshd_fp16_br4_bc4)

MLLM_CPU_ASM_FOOTER
