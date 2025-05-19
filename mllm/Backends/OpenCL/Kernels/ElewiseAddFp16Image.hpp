// This file is the template of OpenCL kernels.
//
// OpenCL File      :elewise_add_fp16_image.cl
// Generate Date    :Mon May 19 10:43:03 2025
//
#pragma once

namespace mllm::opencl {

class ElewiseAddFp16ImageOpenCLKernel : protected OpenCLKernel {
 public:
  void defineMllmOpenclSupportFp16() { build_options_.emplace_back("-DMLLM_OPENCL_SUPPORT_FP16"); }

 private:
  static constexpr const char* source_opencl_code = 
"// </FLAGS START>\n"
"//\n"
"// MLLM_OPENCL_SUPPORT_FP16:def\n"
"//\n"
"// </FLAGS END>\n"
"//\n"
"// Elementwise Op Kernel(image memory)\n"
"// 1. elementwise_add_fp16\n"
"// \n"
"// \n"
"\n"
"#ifdef MLLM_OPENCL_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#pragma OPENCL EXTENSION cl_khr_fp16_arithmetic : enable\n"
"\n"
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | \n"
"                         CLK_ADDRESS_CLAMP_TO_EDGE | \n"
"                         CLK_FILTER_NEAREST;\n"
"\n"
"__attribute__((reqd_work_group_size(64, 1, 1)))\n"
"__kernel void elementwise_add_fp16_adreno(\n"
"    __read_only image2d_t inputA,\n"
"    __read_only image2d_t inputB,\n"
"    __write_only image2d_t output,\n"
"    int width,\n"
"    int height,\n"
"    int aligned_width\n"
") {\n"
"    const int global_id = get_global_id(0);\n"
"    const int group_id = get_group_id(0);\n"
"    const int local_id = get_local_id(0);\n"
"\n"
"    const int VEC_SIZE = 4;\n"
"    const int elements_per_workitem = 4;\n"
"\n"
"    int2 base_pos = (int2)(\n"
"        group_id * (get_local_size(0) * elements_per_workitem) + local_id * VEC_SIZE,\n"
"        get_global_id(1)\n"
"    );\n"
"\n"
"    half4 inA[VEC_SIZE], inB[VEC_SIZE];\n"
"    #pragma unroll\n"
"    for (int i = 0; i < VEC_SIZE; ++i) {\n"
"        int2 pos = base_pos + (int2)(i, 0);\n"
"        if (pos.x < aligned_width && pos.y < height) {\n"
"            inA[i] = read_imageh(inputA, sampler, pos);\n"
"            inB[i] = read_imageh(inputB, sampler, pos);\n"
"        } else {\n"
"            inA[i] = (half4)(0.0h);\n"
"            inB[i] = (half4)(0.0h);\n"
"        }\n"
"    }\n"
"\n"
"    half4 result[VEC_SIZE];\n"
"    #pragma unroll\n"
"    for (int i = 0; i < VEC_SIZE; ++i) {\n"
"        result[i] = inA[i] + inB[i];\n"
"    }\n"
"\n"
"    #pragma unroll\n"
"    for (int i = 0; i < VEC_SIZE; ++i) {\n"
"        int2 write_pos = base_pos + (int2)(i, 0);\n"
"        if (write_pos.x < width && write_pos.y < height) {\n"
"            write_imageh(output, write_pos, result[i]);\n"
"        }\n"
"    }\n"
"}\n"
"#endif\n"
"\n"
;
};

}  // namespace mllm::opencl
