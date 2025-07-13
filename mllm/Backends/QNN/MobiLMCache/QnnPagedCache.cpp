/**
 * @file QnnPagedCache.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/MobiLMCache/QnnPagedCache.hpp"
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/Backends/QNN/QnnAllocator.hpp"
#include "mllm/Engine/Context.hpp"

#include <QNN/QnnTypes.h>
#include <memory>
#include <string>

namespace mllm::qnn {

QnnPagedCache::page_idx_t QnnPagedCache::allocPage(const Tensor::shape_t& shape,
                                                   Tensor::dtype_t dtype,
                                                   const alias_name_t& name) {
  auto ret = assign_id_cnt_++;
  auto page_tensor = Tensor::empty(shape, dtype, kQNN)
                         .setName("QnnPageCache.page." + std::to_string(ret))
                         .setMemType(kQnnAppReadWrite)
                         .alloc();
  auto page_tensor_qnn =
      QnnTensorTransform::instance().transform(page_tensor, QNN_TENSOR_VERSION_2);

  // Register to backend
  auto allocator = std::static_pointer_cast<QnnAllocator>(
      MllmEngineCtx::instance().getBackend(kQNN)->getAllocator());
  allocator->registerQnnTensorToSharedBuffer(page_tensor.ptr<char>(), page_tensor_qnn);

  if (!name.empty()) {
    MLLM_RT_ASSERT_EQ(alias_mapping_.count(name), 0);
    alias_mapping_.insert({name, ret});
  }

  MLLM_RT_ASSERT_EQ(page_mapping_.count(ret), 0);
  page_mapping_.insert({ret, {page_tensor, page_tensor_qnn}});

  return ret;
}

QnnPagedCache::page_t QnnPagedCache::pullPage(QnnPagedCache::page_idx_t id) {
  return page_mapping_.count(id) ? page_mapping_[id]
                                 : QnnPagedCache::page_t(Tensor::nil(), Qnn_Tensor_t{});
}

QnnPagedCache::page_t QnnPagedCache::pullPage(const alias_name_t& name) {
  return alias_mapping_.count(name) ? page_mapping_[alias_mapping_[name]]
                                    : QnnPagedCache::page_t(Tensor::nil(), Qnn_Tensor_t{});
}

}  // namespace mllm::qnn
