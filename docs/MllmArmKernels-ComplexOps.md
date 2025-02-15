# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-15T11:06:37+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_baseline/64 | softmax_baseline/64 | iteration | 3557809 | 193.92395010955607 | 193.408388421076 | ns |
| softmax_baseline/128 | softmax_baseline/128 | iteration | 1837857 | 381.9466547209882 | 380.8825773713624 | ns |
| softmax_baseline/256 | softmax_baseline/256 | iteration | 949619 | 739.0879700020323 | 737.0085076225304 | ns |
| softmax_baseline/512 | softmax_baseline/512 | iteration | 484618 | 1449.5634313600322 | 1445.3733022710674 | ns |
| softmax_baseline/1024 | softmax_baseline/1024 | iteration | 244449 | 2872.024389530587 | 2864.3784265838676 | ns |
| softmax_baseline/2048 | softmax_baseline/2048 | iteration | 122733 | 5721.9834764799025 | 5706.123829776831 | ns |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 9319128 | 75.38405363630947 | 75.19310583565323 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4734070 | 148.25421529127436 | 147.8897373718598 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2457531 | 285.5133746063955 | 284.784530897067 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1267027 | 553.726895319707 | 552.414184543818 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 644727 | 1088.287577535988 | 1085.6675228430004 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 325031 | 2158.39796818919 | 2153.1697776519754 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 145193 | 4835.449629153986 | 4819.835646346584 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 36945 | 18991.926323204258 | 18949.702855596162 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 9468 | 73993.17110264729 | 73812.02334178291 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2306 | 311106.71422474977 | 310124.13356461335 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 584 | 1202853.2568465006 | 1196786.2688356158 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 127 | 5231654.362232373 | 5197977.322834637 | ns |
| softmax_v1_f32_kxk_4_threads/64 | softmax_v1_f32_kxk_4_threads/64 | iteration | 99947 | 6683.7194512885135 | 6666.151860486061 | ns |
| softmax_v1_f32_kxk_4_threads/128 | softmax_v1_f32_kxk_4_threads/128 | iteration | 51839 | 13554.009567976127 | 13520.723393584007 | ns |
| softmax_v1_f32_kxk_4_threads/256 | softmax_v1_f32_kxk_4_threads/256 | iteration | 17811 | 39602.29689556019 | 39503.45825613395 | ns |
| softmax_v1_f32_kxk_4_threads/512 | softmax_v1_f32_kxk_4_threads/512 | iteration | 5105 | 137482.13555184935 | 137047.04524975526 | ns |
| softmax_v1_f32_kxk_4_threads/1024 | softmax_v1_f32_kxk_4_threads/1024 | iteration | 1289 | 544631.3359269612 | 541922.4422032568 | ns |
| softmax_v1_f32_kxk_4_threads/2048 | softmax_v1_f32_kxk_4_threads/2048 | iteration | 294 | 2399678.9965835605 | 2381852.4625850306 | ns |
| softmax_v1_f16/64 | softmax_v1_f16/64 | iteration | 7431388 | 94.1626471418716 | 93.85944348485093 | ns |
| softmax_v1_f16/128 | softmax_v1_f16/128 | iteration | 4103621 | 171.0715390097205 | 170.61516158534167 | ns |
| softmax_v1_f16/256 | softmax_v1_f16/256 | iteration | 2059571 | 340.8772433675798 | 339.966150717795 | ns |
| softmax_v1_f16/512 | softmax_v1_f16/512 | iteration | 1047250 | 670.1118500863674 | 668.3572384817356 | ns |
| softmax_v1_f16/1024 | softmax_v1_f16/1024 | iteration | 531261 | 1321.826158879716 | 1318.5755476121858 | ns |
| softmax_v1_f16/2048 | softmax_v1_f16/2048 | iteration | 267705 | 2621.9100763652546 | 2615.260772118562 | ns |
| softmax_v1_f16_kxk/64 | softmax_v1_f16_kxk/64 | iteration | 123428 | 5687.733091390116 | 5672.995762711851 | ns |
| softmax_v1_f16_kxk/128 | softmax_v1_f16_kxk/128 | iteration | 32001 | 21920.39083193421 | 21865.548389112853 | ns |
| softmax_v1_f16_kxk/256 | softmax_v1_f16_kxk/256 | iteration | 8045 | 87217.64934827464 | 87019.71858297067 | ns |
| softmax_v1_f16_kxk/512 | softmax_v1_f16_kxk/512 | iteration | 2036 | 344932.82366423245 | 343972.99557956855 | ns |
| softmax_v1_f16_kxk/1024 | softmax_v1_f16_kxk/1024 | iteration | 509 | 1371316.2043131888 | 1366586.121807467 | ns |
| softmax_v1_f16_kxk/2048 | softmax_v1_f16_kxk/2048 | iteration | 120 | 5636997.399900186 | 5607946.166666652 | ns |
| softmax_v1_f16_kxk_4_threads/64 | softmax_v1_f16_kxk_4_threads/64 | iteration | 94377 | 6916.307617237803 | 6898.717515920163 | ns |
| softmax_v1_f16_kxk_4_threads/128 | softmax_v1_f16_kxk_4_threads/128 | iteration | 47795 | 14669.416027163277 | 14632.672622659247 | ns |
| softmax_v1_f16_kxk_4_threads/256 | softmax_v1_f16_kxk_4_threads/256 | iteration | 16060 | 43377.883063329464 | 43267.07198007456 | ns |
| softmax_v1_f16_kxk_4_threads/512 | softmax_v1_f16_kxk_4_threads/512 | iteration | 4501 | 155530.48389048828 | 155135.53965785482 | ns |
| softmax_v1_f16_kxk_4_threads/1024 | softmax_v1_f16_kxk_4_threads/1024 | iteration | 1165 | 601348.4884057465 | 599358.8935622314 | ns |
| softmax_v1_f16_kxk_4_threads/2048 | softmax_v1_f16_kxk_4_threads/2048 | iteration | 286 | 2465413.0244601546 | 2452549.898601406 | ns |
| transpose_fp32_bshd2bhsd | transpose_fp32_bshd2bhsd | iteration | 186 | 3641853.1559437755 | 3628680.91397848 | ns |
