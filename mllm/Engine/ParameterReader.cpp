/**
 * @file ParameterReader.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/Utils/Common.hpp"

namespace mllm {

MappedFile::MappedFile(const std::string& filename) {
  fd_ = open(filename.c_str(), O_RDONLY);
  if (fd_ == -1) { MLLM_ERROR_EXIT(kError, "Failed to open file {}.", filename); }

  struct stat sb {};
  if (fstat(fd_, &sb) == -1) {
    close(fd_);
    MLLM_ERROR_EXIT(kError, "Failed stat when open file {}, this file may broken.", filename);
  }
  size_ = sb.st_size;

  mapping_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapping_ == MAP_FAILED) {
    close(fd_);
    MLLM_ERROR_EXIT(kError, "Failed to map file {} to memory space.", filename);
  }
}

MappedFile::~MappedFile() {
  if (mapping_) munmap(mapping_, size_);
  if (fd_ != -1) close(fd_);
}

ParameterLoader::ParameterLoader(const std::string& file_name) { load(file_name); }

void ParameterLoader::load(const std::string& filename) {
  mapped_file_ = std::make_shared<MappedFile>(filename);
  const auto* base_ptr = static_cast<const char*>(mapped_file_->data());

  const auto* head = reinterpret_cast<const ParameterPackHead*>(base_ptr);
  validateHeader(head, mapped_file_->size());

  const auto* descriptors =
      reinterpret_cast<const ParameterDescriptor*>(base_ptr + sizeof(ParameterPackHead));

  for (uint32_t i = 0; i < head->parameter_cnt; ++i) {
    const auto& desc = descriptors[i];
    validateDescriptor(desc, mapped_file_->size());

    auto tensor = createTensor(desc);
    params_.emplace(std::string(desc._param_name_view()), tensor);
  }
}

std::shared_ptr<TensorViewImpl> ParameterLoader::operator[](const std::string& name) {
  if (!params_.count(name)) { MLLM_ERROR_EXIT(kError, "Parameter {} not found.", name); }
  return params_.at(name);
}

std::unordered_map<std::string, std::shared_ptr<TensorViewImpl>>& ParameterLoader::params() {
  return params_;
}

void ParameterLoader::validateHeader(const ParameterPackHead* head, size_t file_size) {
  if (head->magic_number != MLLM_PARAMETER_MAGIC_NUMBER) {
    MLLM_ERROR_EXIT(kError, "Invalid magic number, mllm only support .mllm file for inputs.");
  }

  const size_t required_size =
      sizeof(ParameterPackHead) + head->parameter_cnt * sizeof(ParameterDescriptor);
  if (file_size < required_size) { MLLM_ERROR_EXIT(kError, "Incomplete descriptor section."); }
}

void ParameterLoader::validateDescriptor(const ParameterDescriptor& desc, size_t file_size) {
  if (desc.parameter_offset + desc.parameter_size > file_size) {
    MLLM_ERROR_EXIT(kError, "Parameter data out of bounds.");
  }

  if (desc.shape_len > MLLM_TENSOR_SHAPE_MAX_LEN) {
    MLLM_ERROR_EXIT(kError, "Shape length exceeds maximum.");
  }
}

std::shared_ptr<TensorViewImpl> ParameterLoader::createTensor(const ParameterDescriptor& desc) {
  const void* data_ptr = static_cast<const char*>(mapped_file_->data()) + desc.parameter_offset;

  std::vector<int32_t> shape;
  shape.reserve(desc.shape_len);
  for (int i = 0; i < desc.shape_len; ++i) { shape.emplace_back(desc.shape[i]); }

  auto s = TensorStorage::create(shape, static_cast<DataTypes>(desc.parameter_type), kCPU);
  auto t = TensorViewImpl::create(shape, s);

  s->name_ = std::string(desc._param_name_view());
  s->ptr_ = const_cast<void*>(data_ptr);
  s->mem_type_ = kParams;

  return t;
}

std::shared_ptr<ParameterLoader> load(const std::string& file_path) {
  return std::make_shared<ParameterLoader>(file_path);
}

}  // namespace mllm
