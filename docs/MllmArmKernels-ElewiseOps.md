# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-08T14:47:02+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 580671 | 1165.9409592135485 | 1178.5562099705771 | ns |
| add_f32/128 | add_f32/128 | iteration | 39918 | 17630.670279964328 | 17562.87599579195 | ns |
| add_f32/256 | add_f32/256 | iteration | 10243 | 69137.51885895687 | 68902.31709460536 | ns |
| add_f32/512 | add_f32/512 | iteration | 2602 | 269918.40330014995 | 268989.17486549896 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 584 | 1199037.242732218 | 1193726.6027396983 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 117 | 5613152.659348904 | 5566716.0512820305 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 24385 | 28957.136621457554 | 28860.813655935228 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 9135 | 76711.90756903253 | 76433.55205256319 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 3247 | 215913.57246578665 | 215145.0175546634 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 853 | 822816.518914519 | 819621.634232124 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 214 | 3302091.3744672793 | 3278185.1401869776 | ns |
| add_f16/64 | add_f16/64 | iteration | 825964 | 797.6079366204546 | 802.8028957690532 | ns |
| add_f16/128 | add_f16/128 | iteration | 340261 | 2032.4415375239187 | 2056.911767731103 | ns |
| add_f16/256 | add_f16/256 | iteration | 18285 | 38559.196747441914 | 38425.18146020368 | ns |
| add_f16/512 | add_f16/512 | iteration | 4597 | 152686.30266361477 | 152207.1092016824 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 1089 | 642801.8856638208 | 640201.438016588 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 224 | 3182038.4325068775 | 3158728.4017855963 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 85555 | 8028.05376078612 | 8030.595768800383 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 15179 | 45807.18007755835 | 45631.88227160571 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 5995 | 117375.74038683772 | 116892.12477066812 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 1646 | 426855.50091324776 | 425305.65309849073 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 427 | 1645448.025779753 | 1636327.0796252785 | ns |
