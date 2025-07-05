/**
 * @file OpenCLKernel.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-17
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <CL/opencl.hpp>

namespace mllm::opencl {

class OpenCLKernel {
  struct WorkSizeCargo {
    cl_uint work_dim_ = 1;
    std::vector<size_t> global_work_offset_;
    std::vector<size_t> global_work_size_;
    std::vector<size_t> local_work_size_;
  };

  struct DependenciesCargo {
    std::vector<cl::Event> event_wait_list_;
    std::vector<cl::Event> event_output_list_;
  };

 public:
  using ptr_t = std::shared_ptr<OpenCLKernel>;

  inline cl_int run() {
    if (!command_queue_ || kernel_() == nullptr) { return CL_INVALID_COMMAND_QUEUE; }

    if (global_work_size_.empty()) { return CL_INVALID_GLOBAL_WORK_SIZE; }

    if (work_dim_ < 1 || work_dim_ > 3) { return CL_INVALID_WORK_DIMENSION; }

    cl::NDRange offset;
    if (!global_work_offset_.empty()) {
      switch (work_dim_) {
        case 1: offset = cl::NDRange(global_work_offset_[0]); break;
        case 2: offset = cl::NDRange(global_work_offset_[0], global_work_offset_[1]); break;
        case 3:
          offset =
              cl::NDRange(global_work_offset_[0], global_work_offset_[1], global_work_offset_[2]);
          break;
      }
    }

    cl::NDRange global;
    switch (work_dim_) {
      case 1: global = cl::NDRange(global_work_size_[0]); break;
      case 2: global = cl::NDRange(global_work_size_[0], global_work_size_[1]); break;
      case 3:
        global = cl::NDRange(global_work_size_[0], global_work_size_[1], global_work_size_[2]);
        break;
    }

    cl::NDRange local = cl::NullRange;
    if (!local_work_size_.empty()) {
      switch (work_dim_) {
        case 1: local = cl::NDRange(local_work_size_[0]); break;
        case 2: local = cl::NDRange(local_work_size_[0], local_work_size_[1]); break;
        case 3:
          local = cl::NDRange(local_work_size_[0], local_work_size_[1], local_work_size_[2]);
          break;
      }
    }

    const std::vector<cl::Event>* wait_list =
        event_wait_list_.empty() ? nullptr : &event_wait_list_;

    cl::Event* output_event = event_output_list_.empty() ? nullptr : &event_output_list_.front();

    return command_queue_->enqueueNDRangeKernel(kernel_, offset, global, local, wait_list,
                                                output_event);
  }

  inline cl_int operator()() { return run(); }

  template<typename T>
  inline OpenCLKernel& setArg(cl_uint index, const T& value) {
    kernel_.setArg(index, value);
    return *this;
  }

  inline OpenCLKernel& operator[](const WorkSizeCargo& cargo) {
    work_dim_ = cargo.work_dim_;
    global_work_offset_ = cargo.global_work_offset_;
    global_work_size_ = cargo.global_work_size_;
    local_work_size_ = cargo.local_work_size_;
    return *this;
  }

  inline OpenCLKernel& operator[](const std::shared_ptr<cl::CommandQueue>& command_queue) {
    command_queue_ = command_queue;
    return *this;
  }

  inline OpenCLKernel& operator[](const DependenciesCargo& dependencies_cargo) {
    event_wait_list_ = dependencies_cargo.event_wait_list_;
    event_output_list_ = dependencies_cargo.event_output_list_;
    return *this;
  }

  virtual std::string const openclSource() { return ""; }

  virtual std::string name() { return "<BaseOpenCLKernel>"; };

 protected:
  // On the fly data. This kernel is not thread safe.
  cl_uint work_dim_ = 1;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::vector<size_t> global_work_offset_;
  std::vector<size_t> global_work_size_;
  std::vector<size_t> local_work_size_;
  std::vector<cl::Event> event_wait_list_;
  std::vector<cl::Event> event_output_list_;

  // All options, freezed before online compile by opencl backend
  std::vector<std::string> build_options_;
  cl::Kernel kernel_;
};

}  // namespace mllm::opencl
