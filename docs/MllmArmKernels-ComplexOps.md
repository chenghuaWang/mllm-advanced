# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T15:07:03+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 7869322 | 82.44908328843384 | 81.9575335715072 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4354059 | 161.53224772524956 | 160.49664990759197 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2234004 | 315.4182933414634 | 313.4241169666662 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1147852 | 614.6948813917837 | 610.9150378271762 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 583250 | 1208.7024929160807 | 1201.3019768538365 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 293575 | 2396.2926475226027 | 2381.344406029124 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 132899 | 5312.737093648474 | 5278.8937915258975 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 34049 | 20802.41892569524 | 20652.492554847428 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 8623 | 81633.75449431941 | 81097.17035834395 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2129 | 329909.23907991435 | 327436.90465007036 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 535 | 1302651.3812887247 | 1288680.9028037388 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 118 | 5703197.822016191 | 5639598.41525424 | ns |
| softmax_v1_f32_kxk_threads/64 | softmax_v1_f32_kxk_threads/64 | iteration | 85777 | 7970.224873940083 | 7941.48514170465 | ns |
| softmax_v1_f32_kxk_threads/128 | softmax_v1_f32_kxk_threads/128 | iteration | 27089 | 25745.01332659378 | 25646.38181549708 | ns |
| softmax_v1_f32_kxk_threads/256 | softmax_v1_f32_kxk_threads/256 | iteration | 7189 | 98970.14244094826 | 98552.58589511758 | ns |
| softmax_v1_f32_kxk_threads/512 | softmax_v1_f32_kxk_threads/512 | iteration | 1871 | 375941.20202700293 | 374595.5227151259 | ns |
| softmax_v1_f32_kxk_threads/1024 | softmax_v1_f32_kxk_threads/1024 | iteration | 453 | 1552677.0595905718 | 1543789.6710816815 | ns |
| softmax_v1_f32_kxk_threads/2048 | softmax_v1_f32_kxk_threads/2048 | iteration | 104 | 6492253.105814318 | 6441710.067307691 | ns |
