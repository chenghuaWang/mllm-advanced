file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_OP ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)
file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_PASSES ${CMAKE_CURRENT_LIST_DIR}/Passes/*.cpp)
file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_KERNELS
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.cpp)
file(GLOB_RECURSE MLLM_ARM_BACKEND_ASM_KERNELS
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.S)

# We treat kleidiai as a external lib.
set(KLEIDIAI_BUILD_TESTS OFF)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Vendors/kleidiai EXCLUDE_FROM_ALL)
set(kleidiai_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/Vendors/kleidiai)

set(MLLM_KAI_INCLUDE_DIRS
    ${kleidiai_SOURCE_DIR}/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/)

set(KAI_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S
    # for f32_qai8dxp_qsi4c32
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c
    # for qsi8d32p_qai4c32p
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.c
)

set(KAI_HEADERS
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h
)

set(MOBI_ATTN_ARM ON)
set(MOBI_ATTN_ENABLE_TEST OFF)
set(MOBI_ATTN_ENABLE_BENCHMARK OFF)
set(MOBI_ATTN_ENABLE_EXAMPLE OFF)
set(MOBI_ATTN_ARM_COMPILE_OPTIONS ${MLLM_ARM_BACKEND_COMPILE_OPTIONS})
add_subdirectory(Vendors/MobiAttn)

add_library(
  MllmArmBackend SHARED
  ${MLLM_ARM_BACKEND_SRC_OP}
  ${MLLM_ARM_BACKEND_SRC_PASSES}
  ${MLLM_ARM_BACKEND_SRC_KERNELS}
  ${KAI_SOURCES}
  ${MLLM_ARM_BACKEND_ASM_KERNELS}
  ${KAI_HEADERS}
  ArmAllocator.cpp
  ArmBackend.cpp
  ArmQuantizerHelper.cpp)
target_include_directories(MllmArmBackend PUBLIC ${MLLM_KAI_INCLUDE_DIRS})
target_link_libraries(MllmArmBackend PUBLIC MllmRT -fopenmp mafa2)
target_compile_options(MllmArmBackend PUBLIC ${MLLM_ARM_BACKEND_COMPILE_OPTIONS}
                                             -fopenmp)
