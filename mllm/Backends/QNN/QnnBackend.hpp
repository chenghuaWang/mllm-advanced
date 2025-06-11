/**
 * @file QnnBackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <unordered_map>
#include "mllm/Engine/BackendBase.hpp"
#include "mllm/Backends/QNN/Runtime/QnnIRGraph.hpp"
#include "mllm/Backends/QNN/Runtime/QnnLoader.hpp"

namespace mllm::qnn {

class QnnBackend final : public BackendBase {
 public:
  QnnBackend();

  ~QnnBackend();

  bool initHTPBackend();

  inline QnnFuncSymbols& htpFuncSymbols() { return qnn_htp_func_symbols_; }

  inline QnnBackendDevice& htpBackend() { return qnn_htp_backend_; }

  std::shared_ptr<QnnIRGraph> createQnnGraph(const std::string& name,
                                             const ir::graph::SubGraphOp::self_ptr_t& graph_ir,
                                             const QnnFuncSymbols& qnn_func_symbols,
                                             const QnnBackendDevice& qnn_bk_device);

  std::shared_ptr<QnnIRGraph> getCompiledQnnGraph(const std::string& name);

 private:
  QnnBackendDevice qnn_htp_backend_;
  QnnFuncSymbols qnn_htp_func_symbols_;
  std::unordered_map<std::string, std::shared_ptr<QnnIRGraph>> qnn_graphs_;
};

std::shared_ptr<QnnBackend> createQnnBackend();

}  // namespace mllm::qnn