import os
import torch
import struct
import argparse
import numpy as np
from pathlib import Path
from safetensors import safe_open

MLLM_PARAMETER_MAGIC_NUMBER = 0x519ACE0519ACE000
PARAMETER_NAME_LEN = 256
MLLM_TENSOR_SHAPE_MAX_LEN = 8

"""
enum DataTypes : uint32_t {
  kDataTypes_Start = 0,

  // normal
  kInt4,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFp4,
  kFp8,
  kFp16,
  kFp32,

  // Per Tensor Quantization
  kPT_Start,
  kPTInt4_Sym,
  KPTInt4_Asy,
  kPTInt8_Sym,
  kPTInt8_Asy,
  kPT_End,

  // Per Channel Quantization
  kPC_Start,
  kPCInt4_Sym,
  kPCInt4_Asy,
  kPCInt8_Sym,
  kPCInt8_Asy,
  kPC_End,

  // Group Quantization
  kPG_Start,
  // TODO
  kPG_End,

  kDataTypes_End,
};
"""

TYPE_MAPPING = {
    "BFLOAT16": 9,  # NOTE: Map tp fp32. transform_dtype will transform fp16 to fp32
}


def transform_dtype(data, dtype):
    if dtype == "BFLOAT16":
        return data.to(torch.float32)
    return data


def convert_safetensor(input_path, output_path):
    file_name_with_extension = os.path.basename(input_path)
    model_name = os.path.splitext(file_name_with_extension)[0]
    model_name = model_name.encode("utf-8")[: PARAMETER_NAME_LEN - 1]

    print(f"processing {input_path} ...")

    with safe_open(input_path, framework="pt", device="cpu") as f:
        tensors = []
        header_size = (
            268 + len(f.keys()) * 352  # ParameterPackHead + n * ParameterDescriptor
        )

        data_offset = header_size

        for name in f.keys():
            tensor = f.get_tensor(name)
            dtype = str(tensor.dtype).split(".")[-1]

            if len(tensor.shape) > MLLM_TENSOR_SHAPE_MAX_LEN:
                raise ValueError(f"Tensor {name} shape too long")

            shape = list(tensor.shape)
            shape += [0] * (MLLM_TENSOR_SHAPE_MAX_LEN - len(shape))

            tensor = transform_dtype(tensor, dtype.upper())
            param_size = tensor.nbytes

            tensors.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "numpy": tensor.numpy(),
                    "descriptor": {
                        "parameter_id": len(tensors),
                        "parameter_type": TYPE_MAPPING[dtype.upper()],
                        "parameter_size": param_size,
                        "parameter_offset": data_offset,
                        "shape_len": len(tensor.shape),
                        "shape": shape,
                    },
                }
            )
            data_offset += (param_size + 63) // 64 * 64

        with open(output_path, "wb") as out_f:
            pack_head = struct.pack(
                "<Q256sI",  # little-endian, 8+256+4 bytes
                MLLM_PARAMETER_MAGIC_NUMBER,
                model_name.ljust(PARAMETER_NAME_LEN, b"\x00"),
                len(tensors),
            )
            out_f.write(pack_head)

            for tensor in tensors:
                desc = tensor["descriptor"]
                name_bytes = tensor["name"].encode("utf-8")[: PARAMETER_NAME_LEN - 1]

                print(
                    desc["parameter_id"],
                    tensor["name"],
                    desc["parameter_type"],
                    tensor["numpy"].shape,
                )

                descriptor = struct.pack(
                    "<IIQQQ8Q256s",
                    desc["parameter_id"],
                    desc["parameter_type"],
                    desc["parameter_size"],
                    desc["parameter_offset"],
                    desc["shape_len"],
                    *desc["shape"],
                    name_bytes.ljust(PARAMETER_NAME_LEN, b"\x00"),
                )
                out_f.write(descriptor)

            for tensor in tensors:
                data = tensor["numpy"].tobytes()
                out_f.write(data)
                pad_size = (-len(data)) % 64
                out_f.write(b"\x00" * pad_size)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--format", type=str, default="safetensors")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.format == "safetensors":
        convert_safetensor(input_path, output_path)
    else:
        raise ValueError("Unsupported format")

    print(f"Converted {args.input} to {args.output}")
    print(f"Output structure:")
    print(f" - File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
