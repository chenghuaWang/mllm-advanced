import os
import json
import torch
import struct
import argparse
import numpy as np
from pathlib import Path
from safetensors import safe_open
from typing import List, Dict, Union

MLLM_PARAMETER_MAGIC_NUMBER = 0x519ACE0519ACE000
PARAMETER_NAME_LEN = 256
MLLM_TENSOR_SHAPE_MAX_LEN = 16

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--format", type=str, default="safetensors")
parser.add_argument("--bf16_2_fp32", action="store_true")
parser.add_argument("--bf16_2_fp16", action="store_true")
args = parser.parse_args()

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

  kBF16,

  kDataTypes_End,
};
"""

TYPE_MAPPING = {
    "FLOAT16": 8,
    "FLOAT32": 9,
    "BFLOAT16": 24,
}


class ShardedSafetensorsLoader:
    def __init__(self, index_path: Union[Path, str]):
        self.index_path = Path(index_path)
        self.shards: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.index_path.suffix == ".json":
            with open(self.index_path, "r") as f:
                index = json.load(f)
            self.weight_map: Dict[str, str] = index["weight_map"]
            self.base_dir = self.index_path.parent
        elif self.index_path.suffix == ".safetensors":
            self.weight_map = {}
            self.base_dir = self.index_path.parent
            with safe_open(self.index_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.weight_map[key] = self.index_path.name
        else:
            raise ValueError(f"Unsupported file type: {self.index_path.suffix}")

    def keys(self) -> List[str]:
        return list(self.weight_map.keys())

    def _get_shard_path(self, shard_name: str) -> Path:
        return self.base_dir / shard_name

    def _load_shard(self, shard_name: str):
        shard_path = self._get_shard_path(shard_name)

        if shard_name not in self.shards:
            self.shards[shard_name] = {}
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.shards[shard_name][key] = f.get_tensor(key)

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        if tensor_name not in self.weight_map:
            raise KeyError(f"Tensor '{tensor_name}' not found in weight map")

        shard_name = self.weight_map[tensor_name]
        self._load_shard(shard_name)

        if tensor_name not in self.shards[shard_name]:
            raise KeyError(f"Tensor '{tensor_name}' not found in shard '{shard_name}'")

        return self.shards[shard_name][tensor_name]


def transform_dtype(data, dtype):
    if dtype == "BFLOAT16":
        assert args.bf16_2_fp32 or args.bf16_2_fp16
        if args.bf16_2_fp32:
            return data.to(torch.float32), TYPE_MAPPING["FLOAT32"]
        if args.bf16_2_fp16:
            return data.to(torch.float16), TYPE_MAPPING["FLOAT16"]

    return data, TYPE_MAPPING[dtype.upper()]


def convert_safetensor(input_path: Path, output_path: Path):
    file_name_with_extension = os.path.basename(input_path)
    model_name = os.path.splitext(file_name_with_extension)[0]
    model_name = model_name.encode("utf-8")[: PARAMETER_NAME_LEN - 1]

    print(f"processing {input_path} ...")

    f = ShardedSafetensorsLoader(input_path)

    tensors = []
    header_size = (
        268 + len(f.keys()) * 416  # ParameterPackHead + n * ParameterDescriptor
    )

    data_offset = header_size

    for name in f.keys():
        tensor = f.get_tensor(name)
        dtype = str(tensor.dtype).split(".")[-1]

        if len(tensor.shape) > MLLM_TENSOR_SHAPE_MAX_LEN:
            raise ValueError(f"Tensor {name} shape too long")

        shape = list(tensor.shape)
        shape += [0] * (MLLM_TENSOR_SHAPE_MAX_LEN - len(shape))

        tensor, mllm_type_idx = transform_dtype(tensor, dtype.upper())
        param_size = tensor.nbytes

        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "numpy": tensor.numpy(),
                "descriptor": {
                    "parameter_id": len(tensors),
                    "parameter_type": mllm_type_idx,
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
                "<IIQQQ16Q256s",
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


if __name__ == "__main__":
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.format == "safetensors":
        convert_safetensor(input_path, output_path)
    else:
        raise ValueError("Unsupported format")

    print(f"Converted {args.input} to {args.output}")
    print(f"Output structure:")
    print(f" - File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
