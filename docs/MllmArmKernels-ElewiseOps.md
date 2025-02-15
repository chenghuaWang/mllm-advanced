# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-15T11:05:06+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 582375 | 1165.3119829999266 | 1177.7524876584148 | ns |
| add_f32/128 | add_f32/128 | iteration | 39111 | 18111.64337441598 | 18044.342998134896 | ns |
| add_f32/256 | add_f32/256 | iteration | 10041 | 69023.68870131952 | 68798.38920426405 | ns |
| add_f32/512 | add_f32/512 | iteration | 2594 | 269479.45409236115 | 268533.3585196536 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 575 | 1230306.1606666155 | 1224631.5947826116 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 117 | 5847313.906997442 | 5799400.846153823 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 24250 | 28722.881772747438 | 28622.983381444952 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 9176 | 76838.14174166082 | 76558.29097645376 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 3185 | 218894.524842258 | 218054.95572997778 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 837 | 837606.2790787553 | 834161.7228196252 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 209 | 3380987.856567182 | 3357101.751196245 | ns |
| add_f16/64 | add_f16/64 | iteration | 830536 | 795.9768659455751 | 799.6111487040142 | ns |
| add_f16/128 | add_f16/128 | iteration | 348136 | 1985.0233314166321 | 2010.5765390534862 | ns |
| add_f16/256 | add_f16/256 | iteration | 18065 | 38797.2411866499 | 38662.447218385765 | ns |
| add_f16/512 | add_f16/512 | iteration | 4501 | 155995.94955944837 | 155456.45612091292 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 1075 | 654996.4159413046 | 652315.5534884165 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 227 | 3095871.4492625194 | 3072642.748898417 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 83864 | 8156.7747290453635 | 8162.439342273931 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 15443 | 45695.29230418849 | 45543.37207796689 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 5529 | 126626.59455239832 | 126193.93669744465 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 1656 | 423245.9116392771 | 421592.2427536785 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 423 | 1666223.2313870979 | 1655788.3120568097 | ns |
