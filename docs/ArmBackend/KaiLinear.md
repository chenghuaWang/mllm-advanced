# Linear Ops based on Kleidiai

Kleidiai is a highly optimized lib for NN. This library provides highly optimized linear kernels for various data types and matrix layouts. The kernels leverage offline packing, quantization, and specialized tiling strategies to maximize performance on modern CPU architectures.

Mllm wrapped this lib to provide a convenient interface for developers. 

## Support Ops

### KaiLinear_fp16_fp16_fp16p_mxk_kxn 

Optimized for half-precision (float16) matrix multiplication with KxN layout RHS matrix.

**LHS Dtype:** fp16

**RHS Dtype:** fp16 (offline packed)

**DST Dtype:** fp16

**BIAS Dtype:** fp16 (offline packed)

**Layout:** LHS(MxK), RHS(KxN)

**CLAMP:** True, [-FLOAT_INF, FLOAT_INF]

**Key Methods:**

- `pack_rhs_size(K, N)` → Returns packed buffer size in bytes.

- `pack_rhs_offline(packed, rhs, bias, K, N)` → Packs RHS matrix and bias into a packed buffer. bias can be nullptr.

- `matmul(dst, lhs, packed, M, K, N)` → Performs matrix multiplication

**Usage:**

```c++
KaiLinear_fp16_fp16_fp16p_mxk_kxn kernel;

// Pre-pack RHS matrix and bias
size_t packed_size = kernel.pack_rhs_size(K, N);
float16_t* packed_rhs = allocate_aligned(packed_size);
kernel.pack_rhs_offline(packed_rhs, rhs, bias, K, N);

// Execute matrix multiplication
float16_t* dst = allocate_aligned(M * N * sizeof(float16_t));
kernel.matmul(dst, lhs, packed_rhs, M, K, N);
```

### KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk

Quantized kernel for FP32 inputs with row-major RHS matrix (INT8 LHS + INT4 RHS).

**LHS Dtype:** fp32 (online quant and packed to qai8dxp)

**RHS Dtype:** qsi4c32 (offline packed)

**DST Dtype:** fp32

**BIAS Dtype:** fp32 (offline packed)

**Layout:** LHS(MxK), RHS(NxK)

**CLAMP:** True, [-FLOAT_INF, FLOAT_INF]

**Key Methods:**

- `workspace_size(M, K, tile_cfg)` → Returns workspace size
- `quant_pack_rhs_size(N, K, tile_cfg)` → Returns packed RHS size
- `quant_pack_rhs_offline(packed, rhs, bias, N, K, tile_cfg)` → Quantizes & packs RHS
- `matmul(dst, lhs, packed, workspace, M, K, N, tile_cfg)` → Performs quantized matmul

**Tile Settings:**

- qai8dxp1x8_qsi4c32p4x8_1x4x32
- qai8dxp1x8_qsi4c32p8x8_1x8x32
- qai8dxp4x8_qsi4c32p4x8_8x4x32
- qai8dxp4x8_qsi4c32p4x8_16x4x32
- qai8dxp4x8_qsi4c32p8x8_4x8x32
- qai8dxp1x4_qsi4c32p4x4_1x4

**Usage:**

```c++
KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kernel;
auto tile = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;

// Pack RHS
size_t packed_size = kernel.quant_pack_rhs_size(N, K, tile);
uint8_t* packed_rhs = allocate_aligned(packed_size);
kernel.quant_pack_rhs_offline(packed_rhs, rhs, bias, N, K, tile);

// Prepare workspace
size_t ws_size = kernel.workspace_size(M, K, tile);
void* workspace = allocate_aligned(ws_size);

// Execute quantized matmul
float* dst = allocate_aligned(M * N * sizeof(float));
kernel.matmul(dst, lhs, packed_rhs, workspace, M, K, N, tile);
```

### KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn

Quantized kernel for FP32 inputs with column-major RHS matrix (INT8 LHS + INT4 RHS).

**LHS Dtype:** fp32 (online quant and packed to qai8dxp)

**RHS Dtype:** qsi4c32 (offline packed)

**DST Dtype:** fp32

**BIAS Dtype:** fp32 (offline packed)

**Layout:** LHS(MxK), RHS(KxN)

**CLAMP:** True, [-FLOAT_INF, FLOAT_INF]

**Key Methods:**

- `workspace_size(M, K, tile_cfg)` → Returns workspace size
- `quant_pack_rhs_size(K, N, tile_cfg)` → Returns packed RHS size
- `quant_pack_rhs_offline(packed, rhs, bias, K, N, tile_cfg)` → Quantizes & packs RHS
- `matmul(dst, lhs, packed, workspace, M, K, N, tile_cfg)` → Performs quantized matmul

**Tile Settings:**

- qai8dxp1x8_qsi4c32p4x8_1x4x32
- qai8dxp1x8_qsi4c32p8x8_1x8x32
- qai8dxp4x8_qsi4c32p4x8_8x4x32
- qai8dxp4x8_qsi4c32p4x8_16x4x32
- qai8dxp4x8_qsi4c32p8x8_4x8x32
- qai8dxp1x4_qsi4c32p4x4_1x4

**Usage:**

```c++
KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn kernel;
auto tile = KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;

// Pack RHS
size_t packed_size = kernel.quant_pack_rhs_size(K, N, tile);
uint8_t* packed_rhs = allocate_aligned(packed_size);
kernel.quant_pack_rhs_offline(packed_rhs, rhs, bias, K, N, tile);

// Prepare workspace
size_t ws_size = kernel.workspace_size(M, K, tile);
void* workspace = allocate_aligned(ws_size);

// Execute quantized matmul
float* dst = allocate_aligned(M * N * sizeof(float));
kernel.matmul(dst, lhs, packed_rhs, workspace, M, K, N, tile);
```

## Requirements

- C++17 compatible compiler or higher standard.
- ARMv8+ CPU with FP16 support.

## Limitations

- SME kernels are not supported yet.
