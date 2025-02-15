# MllmArmKernels-GemmOps Benchmark Results

device: xiaomi 12s

data: 2025-02-15T11:07:32+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemmOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| sgemm_v1/64 | sgemm_v1/64 | iteration | 12662 | 55474.25769996313 | 55284.24940767652 | ns |
| sgemm_v1/128 | sgemm_v1/128 | iteration | 1477 | 465114.63981409307 | 463588.08056872035 | ns |
| sgemm_v1/256 | sgemm_v1/256 | iteration | 197 | 3627384.9897587737 | 3612432.081218275 | ns |
| sgemm_v1/512 | sgemm_v1/512 | iteration | 19 | 37473273.0009495 | 37332094.73684212 | ns |
| sgemm_v1/1024 | sgemm_v1/1024 | iteration | 3 | 241156024.33448657 | 239778263.66666684 | ns |
| sgemm_v1/2048 | sgemm_v1/2048 | iteration | 1 | 1558966665.994376 | 1551972299.0 | ns |
| sgemm_v1_4_threads/64 | sgemm_v1_4_threads/64 | iteration | 18179 | 38457.41476427035 | 38328.89201826282 | ns |
| sgemm_v1_4_threads/128 | sgemm_v1_4_threads/128 | iteration | 2498 | 294672.9255403788 | 293645.416733387 | ns |
| sgemm_v1_4_threads/256 | sgemm_v1_4_threads/256 | iteration | 307 | 2310033.5896659815 | 2301029.804560263 | ns |
| sgemm_v1_4_threads/512 | sgemm_v1_4_threads/512 | iteration | 27 | 25254519.66707974 | 25166265.37037036 | ns |
| sgemm_v1_4_threads/1024 | sgemm_v1_4_threads/1024 | iteration | 3 | 262208506.9985284 | 261256651.0000001 | ns |
| sgemm_v1_4_threads/2048 | sgemm_v1_4_threads/2048 | iteration | 1 | 870936978.9867196 | 838983604.000001 | ns |
