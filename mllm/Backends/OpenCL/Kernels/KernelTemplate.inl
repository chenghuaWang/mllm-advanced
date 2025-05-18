// This file is the template of OpenCL kernels.
//
// OpenCL File      :<&opencl_file&>
// Generate Date    :<&generate_date&>
//
#pragma once

namespace mllm::opencl {

class <&opencl_kernel_name&>OpenCLKernel : protected OpenCLKernel {
 public:
  <&opencl_kernel_cfg_funcs&>

 private:
  static constexpr const char* source_opencl_code = 
<&opencl_kernel_source&>
;
};

}  // namespace mllm::opencl
