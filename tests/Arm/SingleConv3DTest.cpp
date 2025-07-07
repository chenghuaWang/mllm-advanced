#include <gtest/gtest.h>

#include "mllm/Core/DataTypes.hpp"

#include "mllm/Nn/Layers/Conv3D.hpp"
#include "mllm/Nn/Module.hpp"
#include "mllm/Engine/Context.hpp"
#include "mllm/Backends/Arm/ArmBackend.hpp"

using namespace mllm;  // NOLINT

Tensor refConv3d(Tensor& activation, Tensor& weights) {
  // Activation shape is: [Batch, Channels, Time, Hight, Weight]
  // Weight shape is: [out_channels, in_channels, kernel_t_size, kernel_h_size, kernel_w_size]
  // Padding is always 0 and dilation is always 1, stride is always kernelsize
  auto batch_size = activation.shape()[0];
  auto in_channels = activation.shape()[1];
  auto out_channels = weights.shape()[0];
  auto in_times = activation.shape()[2];
  auto in_height = activation.shape()[3];
  auto in_width = activation.shape()[4];

  auto kernel_t_size = weights.shape()[2];
  auto kernel_h_size = weights.shape()[3];
  auto kernel_w_size = weights.shape()[4];

  auto out_shape = [](int dim_size, int kernel_size, int stride_size, int padding_size,
                      int dilation_size) -> int32_t {
    // FIXME use floor.
    return ((dim_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size)
           + 1;
  };

  auto out_times = out_shape(in_times, kernel_t_size, kernel_t_size, 0, 1);
  auto out_height = out_shape(in_height, kernel_h_size, kernel_h_size, 0, 1);
  auto out_width = out_shape(in_width, kernel_w_size, kernel_w_size, 0, 1);

  Tensor output =
      Tensor::empty({batch_size, out_channels, out_times, out_height, out_width}, kFp32, kCPU)
          .alloc();

  auto a_ptr = activation.ptr<float>();
  auto w_ptr = weights.ptr<float>();
  auto o_ptr = output.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int ot = 0; ot < out_times; ++ot) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.0f;

            for (int ic = 0; ic < in_channels; ++ic) {
              for (int kt = 0; kt < kernel_t_size; ++kt) {
                for (int kh = 0; kh < kernel_h_size; ++kh) {
                  for (int kw = 0; kw < kernel_w_size; ++kw) {
                    int it = ot * kernel_t_size + kt;
                    int ih = oh * kernel_h_size + kh;
                    int iw = ow * kernel_w_size + kw;

                    int a_idx = b * (in_channels * in_times * in_height * in_width)
                                + ic * (in_times * in_height * in_width)
                                + it * (in_height * in_width) + ih * in_width + iw;

                    int w_idx = oc * (in_channels * kernel_t_size * kernel_h_size * kernel_w_size)
                                + ic * (kernel_t_size * kernel_h_size * kernel_w_size)
                                + kt * (kernel_h_size * kernel_w_size) + kh * kernel_w_size + kw;

                    sum += a_ptr[a_idx] * w_ptr[w_idx];
                  }
                }
              }
            }

            int o_idx = b * (out_channels * out_times * out_height * out_width)
                        + oc * (out_times * out_height * out_width) + ot * (out_height * out_width)
                        + oh * out_width + ow;

            o_ptr[o_idx] = sum;
          }
        }
      }
    }
  }
  return output;
}

class SingleConv3D final : public nn::Module {
 public:
  nn::Conv3D conv3d_;

  SingleConv3D() = default;

  explicit SingleConv3D(const std::string& name) {
    selfAssignName(name);
    conv3d_ = reg<nn::Conv3D>("conv3d", 3, 1280, std::vector<int32_t>{2, 14, 14},
                              std::vector<int32_t>{2, 14, 14}, false, Conv3DOpImplType::kDefault);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {conv3d_(inputs[0])};
  }
};

Tensor singleConv3D(Tensor& activation, Tensor& weights) {
  weights.setMemType(kParams);
  auto net = SingleConv3D("model");

  // make paramloader
  auto ploader = std::make_shared<ParameterLoader>();
  ploader->params().insert({"model.conv3d.weight", weights.impl()});
  net.load(ploader);
  weights.setMemType(kNormal);
  return net(activation)[0];
}

int main(int argc, char** argv) {
  auto& ctx = MllmEngineCtx::instance();
  ctx.registerBackend(mllm::arm::createArmBackend());
  ctx.mem()->initBuddyCtx(kCPU);
  ctx.mem()->initOC(kCPU);

  {
    auto weights = Tensor::random({1280, 3, 2, 14, 14}, -1, 1, kFp32, kCPU);
    auto activation = Tensor::random({1, 3, 2, 196, 196}, -1, 1, kFp32, kCPU);
    auto ref_output = refConv3d(activation, weights);
    auto cal_output = singleConv3D(activation, weights);

    for (int i = 0; i < ref_output.numel(); ++i) {
      auto error = std::abs(ref_output.ptr<float>()[i] - cal_output.ptr<float>()[i]);
      if (error > 0.00001f) {
        Dbg(error, ref_output.ptr<float>()[i], cal_output.ptr<float>()[i]);
        break;
      }
    }
  }

  ctx.shutdown();
}