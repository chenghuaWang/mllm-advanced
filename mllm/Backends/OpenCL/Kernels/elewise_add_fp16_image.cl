// </FLAGS START>
//
// MLLM_OPENCL_SUPPORT_FP16:def
//
// </FLAGS END>
//
// Elementwise Op Kernel(image memory)
// 1. elementwise_add_fp16
// 
// 

#ifdef MLLM_OPENCL_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp16_arithmetic : enable

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                         CLK_ADDRESS_CLAMP_TO_EDGE | 
                         CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void elementwise_add_fp16_adreno(
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    int width,
    int height,
    int aligned_width
) {
    const int global_id = get_global_id(0);
    const int group_id = get_group_id(0);
    const int local_id = get_local_id(0);

    const int VEC_SIZE = 4;
    const int elements_per_workitem = 4;

    int2 base_pos = (int2)(
        group_id * (get_local_size(0) * elements_per_workitem) + local_id * VEC_SIZE,
        get_global_id(1)
    );

    half4 inA[VEC_SIZE], inB[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        int2 pos = base_pos + (int2)(i, 0);
        if (pos.x < aligned_width && pos.y < height) {
            inA[i] = read_imageh(inputA, sampler, pos);
            inB[i] = read_imageh(inputB, sampler, pos);
        } else {
            inA[i] = (half4)(0.0h);
            inB[i] = (half4)(0.0h);
        }
    }

    half4 result[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        result[i] = inA[i] + inB[i];
    }

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        int2 write_pos = base_pos + (int2)(i, 0);
        if (write_pos.x < width && write_pos.y < height) {
            write_imageh(output, write_pos, result[i]);
        }
    }
}
#endif
