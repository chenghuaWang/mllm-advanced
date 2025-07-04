/**
 * @file FillOp.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/Arm/Ops/FillOp.hpp"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Utils/Log.hpp"
#include <arm_neon.h>
#include <random>

namespace mllm::arm {

namespace {
void __iteration_copy_tensor(Tensor& dst, Tensor& src, size_t ele_size,
                             std::vector<int32_t>& indices, std::vector<int32_t>& shape,
                             size_t dim = 0) {
  if (dim == indices.size()) {
    std::memcpy(dst.offsettedRawPtr(indices), src.offsettedRawPtr(indices), ele_size);
    return;
  }

  for (size_t i = 0; i < shape[dim]; ++i) {
    indices[dim] = i;
    __iteration_copy_tensor(dst, src, ele_size, indices, shape, dim + 1);
  }
}
}  // namespace

ArmFillOp::ArmFillOp(const FillOpCargo& cargo) : FillOp(cargo) {}

void ArmFillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& t = inputs[0];
  auto dtype = t.dtype();
  // type
  // 0 -> zeros
  // 1 -> ones
  // 2 -> specific
  // 3 -> random
  // 4 -> arrange
  // 5 -> make tensor contiguous
  switch (cargo_.type) {
    case 0: {
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
      switch (dtype) {
        case kFp32: std::memset(t.ptr<float>(), 0, t.numel() * sizeof(float)); break;
        case kFp16:
          std::fill(t.ptr<float16_t>(), t.ptr<float16_t>() + t.numel(),
                    static_cast<float16_t>(0.f));
          break;
        case kInt32: std::fill(t.ptr<int32_t>(), t.ptr<int32_t>() + t.numel(), 0); break;
        default: NYI("ArmFillOp type=ones, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 1: {
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
      switch (dtype) {
        case kFp32: std::fill(t.ptr<float>(), t.ptr<float>() + t.numel(), 1.f); break;
        case kFp16:
          std::fill(t.ptr<float16_t>(), t.ptr<float16_t>() + t.numel(),
                    static_cast<float16_t>(1.f));
          break;
        case kInt32: std::fill(t.ptr<int32_t>(), t.ptr<int32_t>() + t.numel(), 1); break;
        default: NYI("ArmFillOp type=zeros, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 2: MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid()); break;
    case 3: {
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
      static thread_local std::random_device rd;
      static thread_local uint64_t seed = static_cast<uint64_t>(rd()) << 32 | rd();
      uint64_t state[2] = {seed, seed ^ 0x7263d9bd8409f526};

      auto rand_u64 = [](uint64_t* s) {
        uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        uint64_t result = s0 + s1;

        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);

        return result;
      };

      auto rand_float = [&]() {
        constexpr uint32_t mask = 0x7FFFFF;
        constexpr float norm = 1.0f / (1 << 23);
        uint32_t bits = static_cast<uint32_t>(rand_u64(state)) & mask;
        return bits * norm;
      };

      auto rand_float16 = [&]() {
        constexpr uint16_t mask = 0x7FF;
        constexpr float norm = 1.0f / (1 << 11);
        uint16_t bits = static_cast<uint16_t>(rand_u64(state)) & mask;
        return static_cast<float16_t>(bits * norm);
      };

      auto rand_int32 = [&]() {
        uint64_t r = rand_u64(state);
        uint32_t lo = static_cast<uint32_t>(r);
        uint32_t hi = static_cast<uint32_t>(r >> 32);
        return static_cast<int32_t>((static_cast<uint64_t>(hi) * 101) >> 32);
      };

      switch (dtype) {
        case kFp32: {
          auto ptr = t.ptr<float>();
          const int numel = t.numel();
          const int chunk = numel / 4;
          for (int i = 0; i < chunk; ++i) {
            float32x4_t rand_vec = {rand_float(), rand_float(), rand_float(), rand_float()};
            vst1q_f32(ptr + i * 4, rand_vec);
          }
          for (int i = chunk * 4; i < numel; ++i) { ptr[i] = rand_float(); }
          break;
        }
        case kFp16: {
          auto ptr = t.ptr<float16_t>();
          const int numel = t.numel();
          const int chunk = numel / 8;
          for (int i = 0; i < chunk; ++i) {
            float16x8_t rand_vec = {rand_float16(), rand_float16(), rand_float16(), rand_float16(),
                                    rand_float16(), rand_float16(), rand_float16(), rand_float16()};
            vst1q_f16(reinterpret_cast<float16_t*>(ptr + i * 8), rand_vec);
          }
          for (int i = chunk * 8; i < numel; ++i) { ptr[i] = rand_float16(); }
          break;
        }
        case kInt32: {
          auto ptr = t.ptr<int32_t>();
          const int numel = t.numel();
          const int chunk = numel / 4;
          for (int i = 0; i < chunk; ++i) {
            int32x4_t rand_vec = {rand_int32(), rand_int32(), rand_int32(), rand_int32()};
            vst1q_s32(ptr + i * 4, rand_vec);
          }
          for (int i = chunk * 4; i < numel; ++i) { ptr[i] = rand_int32(); }
          break;
        }
        default: NYI("ArmFillOp type=random, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 4: {
      MLLM_RT_ASSERT_EQ(inputs[0].uuid(), outputs[0].uuid());
      switch (dtype) {
        case kFp32:
          for (int i = 0; i < t.numel(); ++i) {
            (*(t.ptr<float>() + i)) = static_cast<float>(cargo_.start + i * cargo_.step);
          }
          break;
        case kFp16:
          for (int i = 0; i < t.numel(); ++i) {
            (*(t.ptr<float16_t>() + i)) = static_cast<float16_t>(cargo_.start + i * cargo_.step);
          }
          break;
        case kInt32:
          for (int i = 0; i < t.numel(); ++i) {
            (*(t.ptr<int32_t>() + i)) = static_cast<int32_t>(cargo_.start + i * cargo_.step);
          }
          break;
        default: NYI("ArmFillOp type=arange, dtype={}.", dataTypes2Str(dtype));
      }
      break;
    }
    case 5: {
      auto t = inputs[0];
      auto o = outputs[0];
      auto shape = t.shape();
      auto indicies = std::vector<int32_t>(shape.size(), 0);
      __iteration_copy_tensor(o, t, dataTypeSize(t.dtype()), indicies, shape, 0);
      break;
    }
    default:
      MLLM_WARN("ArmFillOp found cargo.type={}, which is not supported yet. The ArmFillOp will do "
                "nothing on input tensor.",
                cargo_.type);
      break;
  }
}

}  // namespace mllm::arm
