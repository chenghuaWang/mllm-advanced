# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-14T15:09:56+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 592889 | 1174.0436491646667 | 1179.653650177366 | ns |
| add_f32/128 | add_f32/128 | iteration | 38811 | 18114.218398883193 | 17973.85225838182 | ns |
| add_f32/256 | add_f32/256 | iteration | 9954 | 70788.91749986759 | 70290.92746634033 | ns |
| add_f32/512 | add_f32/512 | iteration | 2371 | 297614.1384843413 | 294809.0219316637 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 541 | 1290621.7819459343 | 1275921.1700554353 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 108 | 6569567.871877182 | 6490142.851851828 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 23032 | 29965.568914100983 | 29849.431312952347 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 8801 | 79833.37907876565 | 79448.68787637493 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 2953 | 237202.3289525062 | 235907.2963088386 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 788 | 873660.8388847293 | 868114.1992386393 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 191 | 3717732.639234825 | 3684956.382198914 | ns |
| add_f16/64 | add_f16/64 | iteration | 827102 | 810.7300680800537 | 808.0158251339772 | ns |
| add_f16/128 | add_f16/128 | iteration | 341862 | 2027.9414730462966 | 2044.8107072432351 | ns |
| add_f16/256 | add_f16/256 | iteration | 16327 | 42043.243047696924 | 41724.03956637192 | ns |
| add_f16/512 | add_f16/512 | iteration | 3826 | 180965.94961930427 | 179565.8849973395 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 964 | 729793.591488344 | 722458.2551867478 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 198 | 3341876.837745958 | 3306729.6969698933 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 85933 | 8095.160182950409 | 8101.867524692051 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 14936 | 47469.72592790094 | 47221.791778249215 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 5420 | 129427.51175470519 | 128806.43505531266 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 1590 | 443214.6864871453 | 440566.0540880438 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 402 | 1767185.4226833757 | 1752944.3880597528 | ns |
