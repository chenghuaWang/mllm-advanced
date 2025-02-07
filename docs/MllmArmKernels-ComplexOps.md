# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T20:11:02+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_baseline/64 | softmax_baseline/64 | iteration | 2603112 | 270.59750983157613 | 269.05758722636597 | ns |
| softmax_baseline/128 | softmax_baseline/128 | iteration | 1300464 | 541.5809357280217 | 538.4579757686488 | ns |
| softmax_baseline/256 | softmax_baseline/256 | iteration | 655753 | 1076.0888474698984 | 1069.7670632082509 | ns |
| softmax_baseline/512 | softmax_baseline/512 | iteration | 328667 | 2143.1320150616557 | 2130.1046317397254 | ns |
| softmax_baseline/1024 | softmax_baseline/1024 | iteration | 164629 | 4280.690091028808 | 4254.5628959660835 | ns |
| softmax_baseline/2048 | softmax_baseline/2048 | iteration | 82395 | 8546.650913239435 | 8496.014515443903 | ns |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 8541485 | 82.50868157007521 | 82.03136667687181 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4360583 | 161.7436388655039 | 160.78097836917692 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2244308 | 315.12722496281447 | 313.0754874108187 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1151262 | 612.4000592406572 | 608.8129244255431 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 584293 | 1205.4740532402197 | 1198.4579021141774 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 294576 | 2395.1630547166505 | 2379.60735090435 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 132936 | 5303.8041613705755 | 5271.8938511765045 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 34032 | 20713.643570677472 | 20591.23748236952 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 8645 | 81466.55095386723 | 80940.9039907461 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2141 | 329034.1349828872 | 326688.7874824855 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 520 | 1338077.2230728045 | 1323507.103846153 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 118 | 5680806.398262482 | 5615869.406779673 | ns |
| softmax_v1_f32_kxk_4_threads/64 | softmax_v1_f32_kxk_4_threads/64 | iteration | 88632 | 6984.2614631256665 | 6955.306909468377 | ns |
| softmax_v1_f32_kxk_4_threads/128 | softmax_v1_f32_kxk_4_threads/128 | iteration | 47813 | 14773.833497222375 | 14713.191747014373 | ns |
| softmax_v1_f32_kxk_4_threads/256 | softmax_v1_f32_kxk_4_threads/256 | iteration | 16546 | 42314.614045623275 | 42128.501390064026 | ns |
| softmax_v1_f32_kxk_4_threads/512 | softmax_v1_f32_kxk_4_threads/512 | iteration | 4494 | 154593.62483438532 | 153999.64107699113 | ns |
| softmax_v1_f32_kxk_4_threads/1024 | softmax_v1_f32_kxk_4_threads/1024 | iteration | 1166 | 594615.5385851882 | 590822.4373927956 | ns |
| softmax_v1_f32_kxk_4_threads/2048 | softmax_v1_f32_kxk_4_threads/2048 | iteration | 270 | 2598421.1036875085 | 2577543.5777777876 | ns |
