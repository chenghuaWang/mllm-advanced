/**
 * @file Context.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Engine/MemManager.hpp"
#include "mllm/Engine/Thread.hpp"
#include "mllm/Utils/Log.hpp"

namespace mllm {

MllmEngineCtx::MllmEngineCtx() {
  main_thread_ = std::make_shared<MllmEngineThread>();
  main_thread_mem_ = std::make_shared<MemManager>(MemManagerCargo{});
  thread_map_.insert({main_thread_->threadId(), main_thread_});
}

bool MllmEngineCtx::traceMode() const { return trace_mode_; }

void MllmEngineCtx::setTraceMode(bool trace_mode) { trace_mode_ = trace_mode; }

std::shared_ptr<MllmEngineThread> MllmEngineCtx::thisThread() {
  if (!thread_map_.count(std::this_thread::get_id())) {
    MLLM_WARN("This control thread did registered a MllmEngineThread in MllmEngineCtx. The "
              "MllmEngineCtx will automaticly create one for you. But it is recommend to create "
              "MllmEngineThread manually.");
    thread_map_.insert({std::this_thread::get_id(), std::make_shared<MllmEngineThread>()});
  }
  return thread_map_[std::this_thread::get_id()];
}

uint32_t MllmEngineCtx::getUUID() { return custom_uuid_giver_++; }

std::shared_ptr<MemManager> MllmEngineCtx::mem() const { return main_thread_mem_; }

void MllmEngineCtx::registerBackend(const std::shared_ptr<BackendBase>& new_backend) {
  backends_table_.reg(new_backend->deviceType(), new_backend);
  main_thread_mem_->regAllocator(new_backend->deviceType(), new_backend->getAllocator());
}

std::shared_ptr<BackendBase> MllmEngineCtx::getBackend(DeviceTypes device) {
  return backends_table_[device];
}

std::vector<Tensor> MllmEngineCtx::dispatch(const std::string& name,
                                            const std::vector<Tensor>& inputs) {
  // dispatching already registered layer.
  auto op = thisThread()->layer_ops_table[name];

  std::vector<Tensor> outputs;
  op->reshape(inputs, outputs);
  op->setup(inputs, outputs);
  op->forward(inputs, outputs);
  return outputs;
}

std::vector<Tensor> MllmEngineCtx::dispatch(OpType op_type, const BaseOpCargoBase& base_cargo,
                                            const std::vector<Tensor>& inputs) {
  auto op = backends_table_[inputs[0].device()]->createOp(op_type, base_cargo);

  std::vector<Tensor> outputs;
  op->reshape(inputs, outputs);
  op->setup(inputs, outputs);
  op->forward(inputs, outputs);
  return outputs;
}

}  // namespace mllm
