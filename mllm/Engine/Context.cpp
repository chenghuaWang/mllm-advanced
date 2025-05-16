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
#include "mllm/Engine/Dispatcher.hpp"
#include "mllm/Engine/MemManager.hpp"
#include "mllm/Engine/Thread.hpp"
#include "mllm/Utils/Log.hpp"
#include <chrono>
#include <memory>

namespace mllm {

MllmEngineCtx::MllmEngineCtx() : dispatcher_manager_(this) {
  main_thread_ = std::make_shared<MllmEngineThread>();
  main_thread_mem_ = std::make_shared<MemManager>(MemManagerCargo{});
  thread_map_.insert({main_thread_->threadId(), main_thread_});
  dispatcher_manager_.registerDispatcher("cpu:0:eager", std::make_shared<EagerDispatcher>(), 255,
                                         false);
}

bool MllmEngineCtx::traceMode() const { return trace_mode_; }

void MllmEngineCtx::setTraceMode(bool trace_mode) { trace_mode_ = trace_mode; }

std::shared_ptr<MllmEngineThread> MllmEngineCtx::thisThread() {
  if (!thread_map_.count(std::this_thread::get_id())) {
    MLLM_WARN("This control thread did registered a MllmEngineThread in MllmEngineCtx. The "
              "MllmEngineCtx will automatically create one for you. But it is recommend to create "
              "MllmEngineThread manually.");
    thread_map_.insert({std::this_thread::get_id(), std::make_shared<MllmEngineThread>()});
  }
  return thread_map_[std::this_thread::get_id()];
}

std::shared_ptr<MllmEngineThread> MllmEngineCtx::mainThread() { return main_thread_; }

uint32_t MllmEngineCtx::getUUID() {
  uint32_t ret = custom_uuid_giver_;
  custom_uuid_giver_++;
  return ret;
}

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

  if (trace_mode_) {
    op->reshape(inputs, outputs);
    op->setup(inputs, outputs);
    op->trace(ir_context_.get(), inputs, outputs);
  } else {
    if (perf_) {
      auto start = std::chrono::high_resolution_clock::now();
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      MLLM_INFO("perf| Layer: {}, Time:{} ms", name, duration.count());
    } else {
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
    }
  }

  return outputs;
}

std::vector<Tensor> MllmEngineCtx::dispatch(OpType op_type, const BaseOpCargoBase& base_cargo,
                                            const std::vector<Tensor>& inputs) {
  auto op = backends_table_[inputs[0].device()]->createOp(op_type, base_cargo);

  std::vector<Tensor> outputs;

  if (trace_mode_) {
    op->reshape(inputs, outputs);
    op->setup(inputs, outputs);
    op->trace(ir_context_.get(), inputs, outputs);
  } else {
    if (perf_) {
      auto start = std::chrono::high_resolution_clock::now();
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      MLLM_INFO("perf| Op: {}, Time:{} ms", opType2Str(op_type), duration.count());
    } else {
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
    }
  }

  return outputs;
}

std::vector<Tensor> MllmEngineCtx::sendTask2Dispatcher(const std::string& name,
                                                       const std::vector<Tensor>& inputs) {
  // dispatching already registered layer.
  auto op = thisThread()->layer_ops_table[name];

  auto task = std::shared_ptr<Task>();
  task->task_type_ = kDispatchOpTask;
  task->op_ = op;
  task->inputs_ = inputs;

  dispatcher_manager_.sendTask(task);

  return task->outputs_;
}

std::vector<Tensor> MllmEngineCtx::sendTask2Dispatcher(OpType op_type,
                                                       const BaseOpCargoBase& base_cargo,
                                                       const std::vector<Tensor>& inputs) {
  auto op = backends_table_[inputs[0].device()]->createOp(op_type, base_cargo);

  auto task = std::shared_ptr<Task>();
  task->task_type_ = kDispatchOpTask;
  task->op_ = op;
  task->inputs_ = inputs;

  dispatcher_manager_.sendTask(task);

  return task->outputs_;
}

std::vector<Tensor> MllmEngineCtx::sendAsyncTask2DispatcherAndWait(
    const std::string& dispatcher_name, const std::string& name,
    const std::vector<Tensor>& inputs) {
  // dispatching already registered layer.
  auto op = thisThread()->layer_ops_table[name];

  auto task = std::shared_ptr<Task>();
  task->task_type_ = kDispatchOpTask;
  task->op_ = op;
  task->inputs_ = inputs;

  auto ok = dispatcher_manager_.sendAsyncTaskDirectTo(dispatcher_name, task);
  ok.wait();

  return task->outputs_;
}

std::vector<Tensor> MllmEngineCtx::sendAsyncTask2DispatcherAndWait(
    const std::string& dispatcher_name, OpType op_type, const BaseOpCargoBase& base_cargo,
    const std::vector<Tensor>& inputs) {
  auto op = backends_table_[inputs[0].device()]->createOp(op_type, base_cargo);

  auto task = std::shared_ptr<Task>();
  task->task_type_ = kDispatchOpTask;
  task->op_ = op;
  task->inputs_ = inputs;

  auto ok = dispatcher_manager_.sendAsyncTaskDirectTo(dispatcher_name, task);
  ok.wait();

  return task->outputs_;
}

void MllmEngineCtx::shutdown() {
  thread_map_.clear();
  main_thread_.reset();
  main_thread_mem_->clearGlobalTensor();
  main_thread_mem_->report();

  // we should reset before some dynamic lib unload.(such as cuda rt)
  main_thread_mem_.reset();
}

}  // namespace mllm
