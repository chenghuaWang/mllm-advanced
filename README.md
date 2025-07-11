<h1 align="center">
MLLM
</h1>

<h3 align="center">
<span style="color:#2563eb">M</span>obile × <span style="color:#8b5cf6">M</span>ultimodal
</h3>

<p align="center">
Fast and lightweight LLM inference engine for mobile and edge devices
</p>

<p align="center">
| Arm CPU | X86 CPU | Qualcomm NPU(QNN) |
</p>

---

[MLLM-advanced](https://github.com/chenghuaWang/mllm-advanced) is an extension of the [MLLM](https://github.com/UbiquitousLearning/mllm) project and offers more features and functionalities. 

---

# Features

1. **Dynamic-static integrated computation graph** for easy implementation of your algorithms.  
2. **Multi-backend support** for CPU/NPU. Designed for MOBILE devices.
3. A complete graph-level IR with customizable Passes, and a Qnn Lowering Pipeline provided to compile your graph into QNN Graph.  
4. **MobiLMCache!** An edge-side LMCache with cache pagination management.  

# Supported Models

| Model | Quantization Methods | Backends |
| :---: | :---: | :---: |
| DeepSeek-Distill-Qwen-1.5B | W32A32, W4A32 | Arm-CPU |
| Qwen2VL-2B-Instruct | W32A32, W4A32 | Arm-CPU |

# Run Examples

We demonstrate the usage with an example using DeepSeek distill qwen2 1.5B as the model.

```shell
./demo_ds_qwen2 -m {model path} -j {tokenize.json file}
```

# Install

## Build From Source

The following commands have been tested on Linux systems.

```shell
git clone --recursive https://github.com/chenghuaWang/mllm-advanced.git

export ANDROID_NDK_PATH = /path/to/android-ndk

# build
python task.py tasks/android_build.yaml

# push to u device
python task.py tasks/adb_push.yaml
```

## Install From PyPI

```shell
pip install .
```

# Tools

## Model Convertor

Use the following command to convert models:

```shell
python tools/convertor.py --input {safetensors file} --output {output file} --format safetensors
```

## mllm-quantizer

Usage:

```shell
Usage:
 [-h|--help] <FILE> [-s|--show]

Options:
  -h, --help    Show help message
  <FILE>        Input file path
  -s, --show    Show parameters meta data
```

## mllm-tokenize-checker

Usage:

```shell
Usage:
 [-h|--help] [-j|--json] [-m|--merge] [-t|--type] [-i|--input_str]

Options:
  -h, --help    Show help message.
  -j, --json    SentencePiece json file path.
  -m, --merge   Merge file path.
  -t, --type    Model Type.
  -i, --input_str       Input string for testing.
```

Example:

```shell
./mllm-tokenize-checker -j ../mllm-models/DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json -t ds-qwen2 -i "你好"
```

Output:

```shell
Tensor Meta Info
address:  0x7f85e8224000
name:     qwen2-tokenizer-i0
shape:    1x2
device:   kCPU
dtype:    kInt64
[[151646, 108386]]
```