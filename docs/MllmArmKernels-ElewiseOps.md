# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-06T17:08:49+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 582913 | 1201.0653502357623 | 1206.754652924098 | ns |
| add_f32/128 | add_f32/128 | iteration | 40137 | 17546.14116811216 | 17423.8583601157 | ns |
| add_f32/256 | add_f32/256 | iteration | 10296 | 68568.45816588162 | 68143.8596542319 | ns |
| add_f32/512 | add_f32/512 | iteration | 2479 | 283900.90463707614 | 281982.36264622596 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 586 | 1209664.3126094888 | 1197869.3447099202 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 117 | 6008242.922720627 | 5943710.4700855315 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 22795 | 30974.769002593926 | 30791.483044533405 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 6339 | 110902.5967329794 | 110403.80265027039 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 1720 | 402488.1140097467 | 400476.43255813216 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 448 | 1575316.5401162861 | 1566689.7343750747 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 103 | 6525199.154354586 | 6478055.88349505 | ns |
| add_f16/64 | add_f16/64 | iteration | 814278 | 823.1968843171684 | 820.9059817404757 | ns |
| add_f16/128 | add_f16/128 | iteration | 347396 | 1998.5847092761712 | 2017.4163663335976 | ns |
| add_f16/256 | add_f16/256 | iteration | 16847 | 42787.54535663763 | 42515.47225026571 | ns |
| add_f16/512 | add_f16/512 | iteration | 4872 | 145166.21656538665 | 144202.39573072904 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 967 | 705988.91615649 | 700577.5946224405 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 208 | 3383480.7944423608 | 3350479.59134619 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 150021 | 4629.627570173644 | 4633.6211397060815 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 11922 | 58556.824903891444 | 58265.806156675884 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 3415 | 205720.5986425534 | 204757.58008783535 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 897 | 766060.2612450507 | 762206.8472687004 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 217 | 3159135.6781038805 | 3137426.276497895 | ns |
