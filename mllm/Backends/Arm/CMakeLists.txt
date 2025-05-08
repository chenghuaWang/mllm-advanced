file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_OP ${CMAKE_CURRENT_LIST_DIR}/Ops/*.cpp)
file(GLOB_RECURSE MLLM_ARM_BACKEND_SRC_KERNELS
     ${CMAKE_CURRENT_LIST_DIR}/Kernels/*.cpp)

# We treat kleidiai as a external lib.
set(KLEIDIAI_BUILD_TESTS OFF)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/Vendors/kleidiai EXCLUDE_FROM_ALL)
set(kleidiai_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/Vendors/kleidiai)

set(MLLM_KAI_INCLUDE_DIRS
    ${kleidiai_SOURCE_DIR}/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/)

set(KAI_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S
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
  ${MLLM_ARM_BACKEND_SRC_OP} ${MLLM_ARM_BACKEND_SRC_KERNELS} ${KAI_SOURCES}
  ${KAI_HEADERS} ArmAllocator.cpp ArmBackend.cpp)
target_include_directories(MllmArmBackend PRIVATE ${MLLM_KAI_INCLUDE_DIRS})
target_link_libraries(MllmArmBackend PUBLIC MllmRT -fopenmp)
target_link_libraries(MllmArmBackend INTERFACE mafa2)
target_compile_options(MllmArmBackend PUBLIC ${MLLM_ARM_BACKEND_COMPILE_OPTIONS}
                                             -fopenmp)
