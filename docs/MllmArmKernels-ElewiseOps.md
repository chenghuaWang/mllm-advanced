# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T15:05:50+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 592452 | 1176.7687737454912 | 1182.0873623517934 | ns |
| add_f32/128 | add_f32/128 | iteration | 39260 | 17994.457065109764 | 17867.49299541552 | ns |
| add_f32/256 | add_f32/256 | iteration | 10057 | 70440.18300875854 | 69962.3999204556 | ns |
| add_f32/512 | add_f32/512 | iteration | 2598 | 271707.3510574454 | 269083.15242495557 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 561 | 1261308.2330899294 | 1246332.6114081799 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 113 | 6151016.238320668 | 6075812.557522189 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 21651 | 32227.97545843035 | 32052.31684448758 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 6110 | 113677.31620580472 | 113154.49590835698 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 1760 | 396779.0461005775 | 394710.7784091201 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 435 | 1582379.1566607275 | 1573511.2344828064 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 107 | 6693070.167708254 | 6641705.682242895 | ns |
| add_f16/64 | add_f16/64 | iteration | 815064 | 814.9135397935256 | 813.6219082186751 | ns |
| add_f16/128 | add_f16/128 | iteration | 339177 | 2044.3116765655568 | 2062.408235225961 | ns |
| add_f16/256 | add_f16/256 | iteration | 16484 | 42815.21435639215 | 42533.49478282184 | ns |
| add_f16/512 | add_f16/512 | iteration | 4087 | 168328.90325474954 | 166952.24027401768 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 915 | 748614.6525479853 | 741493.4786884721 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 195 | 3608797.2450052174 | 3570520.4461538536 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 149294 | 4735.926885205589 | 4737.1685600255605 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 11912 | 58723.688230132 | 58404.93871726283 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 3307 | 211814.31857384357 | 210811.9570607704 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 857 | 816889.0281172051 | 812662.3757292269 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 217 | 3182292.212158107 | 3158820.5529952655 | ns |
