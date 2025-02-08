# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-08T14:48:32+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_baseline/64 | softmax_baseline/64 | iteration | 2600354 | 269.82978817532575 | 269.1615306992817 | ns |
| softmax_baseline/128 | softmax_baseline/128 | iteration | 1300598 | 538.9950907283471 | 537.7378605841313 | ns |
| softmax_baseline/256 | softmax_baseline/256 | iteration | 656567 | 1069.1452966694017 | 1066.7030645768064 | ns |
| softmax_baseline/512 | softmax_baseline/512 | iteration | 331112 | 2123.549143496579 | 2118.783016018749 | ns |
| softmax_baseline/1024 | softmax_baseline/1024 | iteration | 164668 | 4260.0632424004325 | 4250.313576408289 | ns |
| softmax_baseline/2048 | softmax_baseline/2048 | iteration | 82403 | 8514.333385991631 | 8494.947708214508 | ns |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 8554429 | 82.02644805414764 | 81.82449933245107 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4365984 | 160.7177905348828 | 160.3316013068303 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2231520 | 314.4363666896462 | 313.6928062486559 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1146780 | 612.0746786635845 | 610.6153874326375 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 582639 | 1204.1823393271738 | 1201.3176821325046 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 293788 | 2388.1766988411496 | 2382.678809209364 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 133203 | 5264.920287109638 | 5252.402596037625 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 34060 | 20591.280387660638 | 20549.85543159128 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 8649 | 81195.42351718215 | 81004.84310324892 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2090 | 331460.152153632 | 330651.06698564603 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 529 | 1326926.0982892145 | 1320513.1776937584 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 118 | 5572659.33901486 | 5539271.322033896 | ns |
| softmax_v1_f32_kxk_4_threads/64 | softmax_v1_f32_kxk_4_threads/64 | iteration | 102727 | 6804.6182601285445 | 6784.62653440673 | ns |
| softmax_v1_f32_kxk_4_threads/128 | softmax_v1_f32_kxk_4_threads/128 | iteration | 48255 | 14580.097482157089 | 14538.944461713785 | ns |
| softmax_v1_f32_kxk_4_threads/256 | softmax_v1_f32_kxk_4_threads/256 | iteration | 16817 | 41880.49729400406 | 41751.368496164694 | ns |
| softmax_v1_f32_kxk_4_threads/512 | softmax_v1_f32_kxk_4_threads/512 | iteration | 4619 | 153059.36306638495 | 152594.12080536882 | ns |
| softmax_v1_f32_kxk_4_threads/1024 | softmax_v1_f32_kxk_4_threads/1024 | iteration | 1164 | 593214.847939669 | 590451.664089347 | ns |
| softmax_v1_f32_kxk_4_threads/2048 | softmax_v1_f32_kxk_4_threads/2048 | iteration | 277 | 2557565.0577573776 | 2539043.3393501816 | ns |
| softmax_v1_f16/64 | softmax_v1_f16/64 | iteration | 6766983 | 98.23885725812862 | 97.98189074806322 | ns |
| softmax_v1_f16/128 | softmax_v1_f16/128 | iteration | 3730486 | 188.04570637587207 | 187.62715501411833 | ns |
| softmax_v1_f16/256 | softmax_v1_f16/256 | iteration | 1877455 | 373.78780903140836 | 372.8948139902157 | ns |
| softmax_v1_f16/512 | softmax_v1_f16/512 | iteration | 953266 | 736.0181670079282 | 734.2590210917008 | ns |
| softmax_v1_f16/1024 | softmax_v1_f16/1024 | iteration | 483018 | 1459.1488433327809 | 1455.310120947869 | ns |
| softmax_v1_f16/2048 | softmax_v1_f16/2048 | iteration | 243298 | 2884.270265281409 | 2877.099519930284 | ns |
| softmax_v1_f16_kxk/64 | softmax_v1_f16_kxk/64 | iteration | 112233 | 6253.314345991737 | 6237.917314871765 | ns |
| softmax_v1_f16_kxk/128 | softmax_v1_f16_kxk/128 | iteration | 29112 | 24098.11136294868 | 24043.856382247963 | ns |
| softmax_v1_f16_kxk/256 | softmax_v1_f16_kxk/256 | iteration | 7335 | 95614.36318933021 | 95413.01022494915 | ns |
| softmax_v1_f16_kxk/512 | softmax_v1_f16_kxk/512 | iteration | 1855 | 378301.74609396176 | 377284.20161725144 | ns |
| softmax_v1_f16_kxk/1024 | softmax_v1_f16_kxk/1024 | iteration | 464 | 1507492.5905255128 | 1502115.2392241398 | ns |
| softmax_v1_f16_kxk/2048 | softmax_v1_f16_kxk/2048 | iteration | 110 | 6189199.327358934 | 6158168.118181836 | ns |
| softmax_v1_f16_kxk_4_threads/64 | softmax_v1_f16_kxk_4_threads/64 | iteration | 98244 | 7148.256717907271 | 7124.596453727513 | ns |
| softmax_v1_f16_kxk_4_threads/128 | softmax_v1_f16_kxk_4_threads/128 | iteration | 45438 | 15492.700911085421 | 15445.53169153574 | ns |
| softmax_v1_f16_kxk_4_threads/256 | softmax_v1_f16_kxk_4_threads/256 | iteration | 14830 | 47360.74470661817 | 47213.93108563708 | ns |
| softmax_v1_f16_kxk_4_threads/512 | softmax_v1_f16_kxk_4_threads/512 | iteration | 4152 | 169429.15317836517 | 168970.02649325528 | ns |
| softmax_v1_f16_kxk_4_threads/1024 | softmax_v1_f16_kxk_4_threads/1024 | iteration | 1062 | 653900.0706274059 | 651757.2542372909 | ns |
| softmax_v1_f16_kxk_4_threads/2048 | softmax_v1_f16_kxk_4_threads/2048 | iteration | 266 | 2647116.812010807 | 2633459.199248119 | ns |
