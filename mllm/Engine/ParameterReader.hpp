/**
 * @file ParameterReader.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "mllm/Core/Tensor.hpp"
#include "mllm/Core/TensorImpl.hpp"

namespace mllm {

#define PARAMETER_NAME_LEN 256
#define MLLM_PARAMETER_MAGIC_NUMBER 0x519ACE0519ACE000

// pack to 1B if possible
struct __attribute__((packed)) ParameterDescriptor {
  uint32_t parameter_id;                    // 4B
  uint32_t parameter_type;                  // 4B
  size_t parameter_size;                    // 8B
  size_t parameter_offset;                  // 8B data_ptr = file_begin + parameter_offset
  size_t shape_len;                         // 8B
  size_t shape[MLLM_TENSOR_SHAPE_MAX_LEN];  // 64B
  char name[PARAMETER_NAME_LEN];            // 256B

  [[nodiscard]] std::string_view _param_name_view() const noexcept {
    return {name, strnlen(name, sizeof(name))};
  }
};  // 352B in total of ParameterDescriptor

// pack to 1B if possible
struct __attribute__((packed)) ParameterPackHead {
  uint64_t magic_number;                // 8B
  char model_name[PARAMETER_NAME_LEN];  // 256B
  uint32_t parameter_cnt;               // 4B

  [[nodiscard]] std::string_view _model_name_view() const noexcept {
    return {model_name, strnlen(model_name, sizeof(model_name))};
  }
};  // 268B in total of ParameterDescriptor

static_assert(sizeof(ParameterPackHead) == 268, "Invalid ParameterPackHead size");
static_assert(sizeof(ParameterDescriptor) == 352, "Invalid ParameterDescriptor size");

// ParameterPackHead
// ParameterDescriptor-0
// ParameterDescriptor-1
// ...
// ParameterDescriptor-n
// ParameterData-0
// ParameterData-1
// ...
// ParameterData-n

class MappedFile {
 public:
  explicit MappedFile(const std::string& filename);
  ~MappedFile();

  [[nodiscard]] inline void* data() const { return mapping_; }

  [[nodiscard]] inline size_t size() const { return size_; }

 private:
  void* mapping_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
};

class ParameterLoader {
 public:
  explicit ParameterLoader(const std::string& file_name);

  std::shared_ptr<TensorViewImpl> operator[](const std::string& name);

  std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>>& params();

 private:
  std::shared_ptr<MappedFile> mapped_file_;
  std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>> params_;

  void load(const std::string& filename);

  void validateHeader(const ParameterPackHead* head, size_t file_size);

  void validateDescriptor(const ParameterDescriptor& desc, size_t file_size);

  std::shared_ptr<TensorViewImpl> createTensor(const ParameterDescriptor& desc);
};

std::shared_ptr<ParameterLoader> load(const std::string& file_path);

class ParameterWriter {
 public:
  ParameterWriter() = default;

  void addParams(const std::unordered_map<std::string, Tensor>& params);

  inline void setModelName(const std::string& model_name) { model_name_ = model_name; }

  void write(const std::string& filename);

 private:
  std::string model_name_;
  std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>> params_;
};

void write(const std::string& file_path, const std::unordered_map<std::string, Tensor>& params,
           const std::string& model_name);

void write(const std::string& file_path,
           const std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>>& params,
           const std::string& model_name);
}  // namespace mllm
