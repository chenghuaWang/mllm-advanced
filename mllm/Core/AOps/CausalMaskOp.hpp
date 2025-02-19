/**
 * @file CausalMaskOp.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Core/AOps/BaseOp.hpp"

namespace mllm {
struct CausalMaskOpCargo : public BaseOpCargo<CausalMaskOpCargo> {};

class CausalMaskOp : public BaseOp {
 public:
  explicit CausalMaskOp(const CausalMaskOpCargo& cargo);

  void load(std::shared_ptr<ParameterLoader>& ploader) override;

  void trace(void* trace_context, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  CausalMaskOpCargo cargo_;
};

}  // namespace mllm
