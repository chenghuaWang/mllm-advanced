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
//
// This kernel will be used in:
// 1. Flash Attention 2
// 2. Sliding window FA2

#include "mllm/Utils/AssemblyHelper.hpp"

MLLM_CPU_ASM_EXPORT(fa2_mma0_bshd_fp16_br4_bc4_neon_asm_micro_kernel)
MLLM_CPU_ASM_FUNCTION(fa2_mma0_bshd_fp16_br4_bc4_neon_asm_micro_kernel)
  lsl w10, w4, #1
  lsl w8, w5, #1
  subs w13, w3, #16
  add w11, w10, w4
  add w9, w8, w5
  b.ge .LBB0_2
  movi v0.2d, #0000000000000000
  movi v1.2d, #0000000000000000
  mov w12, wzr
  movi v2.2d, #0000000000000000
  movi v3.2d, #0000000000000000
  movi v4.2d, #0000000000000000
  movi v5.2d, #0000000000000000
  movi v6.2d, #0000000000000000
  movi v7.2d, #0000000000000000
  movi v16.2d, #0000000000000000
  movi v17.2d, #0000000000000000
  movi v18.2d, #0000000000000000
  movi v19.2d, #0000000000000000
  movi v20.2d, #0000000000000000
  movi v21.2d, #0000000000000000
  movi v23.2d, #0000000000000000
  movi v22.2d, #0000000000000000
  b .LBB0_5
.LBB0_2:
  stp d13, d12, [sp, #-128]!
  stp d11, d10, [sp, #16]
  stp d9, d8, [sp, #32]
  str x27, [sp, #48]
  stp x26, x25, [sp, #64]
  stp x24, x23, [sp, #80]
  stp x22, x21, [sp, #96]
  stp x20, x19, [sp, #112]
  movi v22.2d, #0000000000000000
  movi v23.2d, #0000000000000000
  add x15, x0, w4, sxtw #1
  movi v21.2d, #0000000000000000
  movi v20.2d, #0000000000000000
  add x16, x0, w10, sxtw #1
  movi v19.2d, #0000000000000000
  movi v18.2d, #0000000000000000
  add x17, x0, w11, sxtw #1
  movi v17.2d, #0000000000000000
  movi v16.2d, #0000000000000000
  add x18, x1, w5, sxtw #1
  movi v7.2d, #0000000000000000
  movi v6.2d, #0000000000000000
  add x7, x1, w8, sxtw #1
  movi v5.2d, #0000000000000000
  movi v4.2d, #0000000000000000
  add x19, x1, w9, sxtw #1
  movi v3.2d, #0000000000000000
  movi v2.2d, #0000000000000000
  mov x14, xzr
  movi v1.2d, #0000000000000000
  movi v0.2d, #0000000000000000
  mov x12, xzr
.LBB0_3:
  add x22, x0, x14
  add x23, x15, x14
  add x24, x16, x14
  add x25, x17, x14
  add x26, x1, x14
  add x27, x18, x14
  add x21, x7, x14
  add x20, x19, x14
  prfm pldl1keep, [x22, #256]
  prfm pldl1keep, [x23, #256]
  prfm pldl1keep, [x24, #256]
  prfm pldl1keep, [x25, #256]
  prfm pldl1keep, [x26, #256]
  prfm pldl1keep, [x27, #256]
  prfm pldl1keep, [x21, #256]
  prfm pldl1keep, [x20, #256]
  ldp q28, q24, [x22]
  ldp q29, q25, [x23]
  add x12, x12, #16
  ldp q8, q10, [x26]
  cmp x12, x13
  ldp q30, q26, [x24]
  add x14, x14, #32
  ldp q31, q27, [x25]
  fmlal v22.4s, v28.4h, v8.4h
  fmlal v19.4s, v29.4h, v8.4h
  fmlal v7.4s, v30.4h, v8.4h
  ldp q13, q12, [x20]
  fmlal v3.4s, v31.4h, v8.4h
  fmlal2 v22.4s, v28.4h, v8.4h
  fmlal2 v19.4s, v29.4h, v8.4h
  fmlal v20.4s, v28.4h, v13.4h
  fmlal2 v7.4s, v30.4h, v8.4h
  fmlal v16.4s, v29.4h, v13.4h
  fmlal v4.4s, v30.4h, v13.4h
  fmlal2 v3.4s, v31.4h, v8.4h
  ldp q9, q8, [x27]
  fmlal v0.4s, v31.4h, v13.4h
  fmlal v22.4s, v24.4h, v10.4h
  fmlal v19.4s, v25.4h, v10.4h
  fmlal2 v20.4s, v28.4h, v13.4h
  fmlal v7.4s, v26.4h, v10.4h
  fmlal v23.4s, v28.4h, v9.4h
  fmlal v18.4s, v29.4h, v9.4h
  fmlal v3.4s, v27.4h, v10.4h
  fmlal v6.4s, v30.4h, v9.4h
  fmlal v2.4s, v31.4h, v9.4h
  fmlal2 v16.4s, v29.4h, v13.4h
  fmlal2 v4.4s, v30.4h, v13.4h
  fmlal2 v0.4s, v31.4h, v13.4h
  fmlal2 v22.4s, v24.4h, v10.4h
  fmlal2 v19.4s, v25.4h, v10.4h
  fmlal v20.4s, v24.4h, v12.4h
  fmlal2 v7.4s, v26.4h, v10.4h
  fmlal2 v23.4s, v28.4h, v9.4h
  fmlal2 v18.4s, v29.4h, v9.4h
  fmlal2 v3.4s, v27.4h, v10.4h
  ldp q11, q10, [x21]
  fmlal2 v6.4s, v30.4h, v9.4h
  fmlal2 v2.4s, v31.4h, v9.4h
  fmlal v16.4s, v25.4h, v12.4h
  fmlal v4.4s, v26.4h, v12.4h
  fmlal v0.4s, v27.4h, v12.4h
  fmlal2 v20.4s, v24.4h, v12.4h
  fmlal v21.4s, v28.4h, v11.4h
  fmlal v17.4s, v29.4h, v11.4h
  fmlal v5.4s, v30.4h, v11.4h
  fmlal v1.4s, v31.4h, v11.4h
  fmlal v23.4s, v24.4h, v8.4h
  fmlal v18.4s, v25.4h, v8.4h
  fmlal v6.4s, v26.4h, v8.4h
  fmlal v2.4s, v27.4h, v8.4h
  fmlal2 v16.4s, v25.4h, v12.4h
  fmlal2 v4.4s, v26.4h, v12.4h
  fmlal2 v0.4s, v27.4h, v12.4h
  fmlal2 v21.4s, v28.4h, v11.4h
  fmlal2 v17.4s, v29.4h, v11.4h
  fmlal2 v5.4s, v30.4h, v11.4h
  fmlal2 v1.4s, v31.4h, v11.4h
  fmlal2 v23.4s, v24.4h, v8.4h
  fmlal2 v18.4s, v25.4h, v8.4h
  fmlal2 v6.4s, v26.4h, v8.4h
  fmlal2 v2.4s, v27.4h, v8.4h
  fmlal v21.4s, v24.4h, v10.4h
  fmlal v17.4s, v25.4h, v10.4h
  fmlal v5.4s, v26.4h, v10.4h
  fmlal v1.4s, v27.4h, v10.4h
  fmlal2 v21.4s, v24.4h, v10.4h
  fmlal2 v17.4s, v25.4h, v10.4h
  fmlal2 v5.4s, v26.4h, v10.4h
  fmlal2 v1.4s, v27.4h, v10.4h
  b.ls .LBB0_3
  ldp x20, x19, [sp, #112]
  ldr x27, [sp, #48]
  ldp x22, x21, [sp, #96]
  ldp x24, x23, [sp, #80]
  ldp x26, x25, [sp, #64]
  ldp d9, d8, [sp, #32]
  ldp d11, d10, [sp, #16]
  ldp d13, d12, [sp], #128
.LBB0_5:
  faddp v22.4s, v22.4s, v22.4s
  faddp v23.4s, v23.4s, v23.4s
  mov w15, #1
  faddp v21.4s, v21.4s, v21.4s
  faddp v20.4s, v20.4s, v20.4s
  add x13, x2, w6, sxtw #2
  faddp v19.4s, v19.4s, v19.4s
  faddp v18.4s, v18.4s, v18.4s
  add x14, x2, w6, sxtw #3
  faddp v17.4s, v17.4s, v17.4s
  faddp v16.4s, v16.4s, v16.4s
  orr w15, w15, w6, lsl #1
  faddp v7.4s, v7.4s, v7.4s
  faddp v6.4s, v6.4s, v6.4s
  mov w16, #12
  faddp v5.4s, v5.4s, v5.4s
  faddp v4.4s, v4.4s, v4.4s
  cmp w12, w3
  faddp v3.4s, v3.4s, v3.4s
  faddp v2.4s, v2.4s, v2.4s
  faddp v1.4s, v1.4s, v1.4s
  faddp v0.4s, v0.4s, v0.4s
  faddp s22, v22.2s
  faddp s23, v23.2s
  faddp s21, v21.2s
  faddp s20, v20.2s
  faddp s19, v19.2s
  faddp s18, v18.2s
  faddp s17, v17.2s
  faddp s16, v16.2s
  faddp s7, v7.2s
  faddp s6, v6.2s
  faddp s5, v5.2s
  stp s22, s23, [x2]
  faddp s4, v4.2s
  faddp s3, v3.2s
  stp s21, s20, [x2, #8]
  faddp s2, v2.2s
  faddp s1, v1.2s
  stp s19, s18, [x13]
  faddp s0, v0.2s
  stp s17, s16, [x13, #8]
  str s7, [x14]
  str s6, [x2, w15, sxtw #2]
  smaddl x15, w6, w16, x2
  stp s5, s4, [x14, #8]
  stp s3, s2, [x15]
  stp s1, s0, [x15, #8]
  b.ge .LBB0_8
  mov w18, w12
  sxtw x17, w4
  sxtw x10, w10
  sxtw x11, w11
  sxtw x4, w5
  sxtw x8, w8
  ubfiz x5, x18, #1, #32
  sxtw x9, w9
  mov w6, w12
  mov w3, w3
  mov x16, xzr
  add x11, x5, x11, lsl #1
  add x10, x5, x10, lsl #1
  add x12, x5, x17, lsl #1
  add x17, x5, x9, lsl #1
  add x18, x5, x8, lsl #1
  add x4, x5, x4, lsl #1
  add x8, x0, x11
  add x9, x0, x10
  add x10, x0, x12
  add x11, x1, x17
  add x12, x1, x18
  add x17, x1, x4
  add x18, x1, x5
  add x0, x0, x5
  sub x1, x3, x6
.LBB0_7:
  lsl x3, x16, #1
  ldr q2, [x2]
  add x16, x16, #1
  cmp x1, x16
  ldr h0, [x18, x3]
  add x4, x17, x3
  ldr h1, [x0, x3]
  ld1 { v0.h }[1], [x4]
  add x4, x12, x3
  fcvt s1, h1
  ld1 { v0.h }[2], [x4]
  add x4, x11, x3
  ld1 { v0.h }[3], [x4]
  fcvtl v0.4s, v0.4h
  fmla v2.4s, v0.4s, v1.s[0]
  ldr h1, [x10, x3]
  fcvt s1, h1
  str q2, [x2]
  ldr q2, [x13]
  fmla v2.4s, v0.4s, v1.s[0]
  ldr h1, [x9, x3]
  fcvt s1, h1
  str q2, [x13]
  ldr q2, [x14]
  fmla v2.4s, v0.4s, v1.s[0]
  ldr h1, [x8, x3]
  fcvt s1, h1
  str q2, [x14]
  ldr q2, [x15]
  fmla v2.4s, v0.4s, v1.s[0]
  str q2, [x15]
  b.ne .LBB0_7
.LBB0_8:
  ret
MLLM_CPU_ASM_FOOTER
