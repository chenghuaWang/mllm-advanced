// This file is the template of OpenCL kernels.
//
// OpenCL File      :<&opencl_file&>
// Generate Date    :<&generate_date&>
//
#pragma once

#include <string>

namespace mllm::opencl {

// clang-format off
class<&opencl_kernel_name&>OpenCLKernel : protected OpenCLKernel {
 public:
  <&opencl_kernel_cfg_funcs&>

  inline std::string const openclSource() override {
    return std::string(source_opencl_code);
  }

  inline std::string name() override { 
    return "<&opencl_kernel_name&>"; 
  };

 private:
  static constexpr const char* source_opencl_code = 
<&opencl_kernel_source&>
;
  // clang-format on
};

}  // namespace mllm::opencl
