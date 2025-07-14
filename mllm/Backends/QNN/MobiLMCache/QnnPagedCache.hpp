/**
 * @file QnnPagedCache.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <QNN/QnnTypes.h>

#include "mllm/Core/Tensor.hpp"

namespace mllm::mobi_lmcache {

class QnnPagedCache {
 public:
  using page_idx_t = int32_t;
  using alias_name_t = std::string;
  using page_t = std::pair<Tensor, Qnn_Tensor_t>;

  page_idx_t allocPage(const Tensor::shape_t& shape, Tensor::dtype_t dtype,
                       const alias_name_t& name = "");

  page_t pullPage(page_idx_t id);

  page_t pullPage(const alias_name_t& name);

 private:
  page_idx_t assign_id_cnt_ = 0;
  std::unordered_map<alias_name_t, page_idx_t> alias_mapping_;
  std::unordered_map<page_idx_t, page_t> page_mapping_;
};

}  // namespace mllm::mobi_lmcache
