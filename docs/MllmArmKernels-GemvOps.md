# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-08T14:47:43+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2759609 | 254.1384899088745 | 253.46312756626028 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 281108 | 2511.0099996876047 | 2505.73737495909 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 74479 | 9304.635508066816 | 9285.120060688247 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 24263 | 27716.88760644117 | 27657.02246218523 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7808 | 89302.50499465504 | 89115.49705430327 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1615 | 433852.263775311 | 432913.11331269366 | ns |
| hgemv_v1_4_threads/64 | hgemv_v1_4_threads/64 | iteration | 95779 | 6871.838962566544 | 6849.14934380187 | ns |
| hgemv_v1_4_threads/128 | hgemv_v1_4_threads/128 | iteration | 61309 | 11862.13484170354 | 11827.853887683701 | ns |
| hgemv_v1_4_threads/256 | hgemv_v1_4_threads/256 | iteration | 35914 | 19983.69800082443 | 19927.970401514754 | ns |
| hgemv_v1_4_threads/512 | hgemv_v1_4_threads/512 | iteration | 19029 | 35008.69562291216 | 34904.78070313729 | ns |
| hgemv_v1_4_threads/1024 | hgemv_v1_4_threads/1024 | iteration | 7713 | 90480.4725790029 | 90225.61584338128 | ns |
| hgemv_v1_4_threads/2048 | hgemv_v1_4_threads/2048 | iteration | 2150 | 325373.23162825993 | 324440.4790697678 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1573820 | 436.58794080277397 | 435.53516285216824 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 246286 | 2801.055606083494 | 2795.197286894094 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 68948 | 10013.59341831395 | 9992.427118988235 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19731 | 35899.54949047808 | 35821.998479549955 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 5897 | 120034.83398215682 | 119768.41241309128 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1326 | 525382.6508332405 | 524239.60935143207 | ns |
| hgemv_v2_hp_4_threads/64 | hgemv_v2_hp_4_threads/64 | iteration | 100994 | 6863.866912959456 | 6843.30210705588 | ns |
| hgemv_v2_hp_4_threads/128 | hgemv_v2_hp_4_threads/128 | iteration | 108891 | 6434.945551025028 | 6414.2903086573115 | ns |
| hgemv_v2_hp_4_threads/256 | hgemv_v2_hp_4_threads/256 | iteration | 67884 | 10235.15759220614 | 10202.330519710122 | ns |
| hgemv_v2_hp_4_threads/512 | hgemv_v2_hp_4_threads/512 | iteration | 28473 | 24749.228777968197 | 24672.161907772166 | ns |
| hgemv_v2_hp_4_threads/1024 | hgemv_v2_hp_4_threads/1024 | iteration | 6739 | 104626.78364640227 | 104344.38017510009 | ns |
| hgemv_v2_hp_4_threads/2048 | hgemv_v2_hp_4_threads/2048 | iteration | 1697 | 410793.95639131963 | 409669.82380671694 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 593280 | 1118.5128472221988 | 1115.8990358683914 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 155036 | 4528.436808174308 | 4518.979965943395 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49823 | 14078.496216710146 | 14048.043594323903 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 14474 | 48370.06584192509 | 48263.09631062606 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3175 | 220818.27402012201 | 220345.9017322835 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 930 | 741179.2666697624 | 739459.9591397839 | ns |
| sgemv_v1_4_threads/64 | sgemv_v1_4_threads/64 | iteration | 94680 | 7360.842659451535 | 7340.482446134361 | ns |
| sgemv_v1_4_threads/128 | sgemv_v1_4_threads/128 | iteration | 61032 | 11315.070880865802 | 11283.736613579735 | ns |
| sgemv_v1_4_threads/256 | sgemv_v1_4_threads/256 | iteration | 34252 | 20604.602767759126 | 20550.01521079061 | ns |
| sgemv_v1_4_threads/512 | sgemv_v1_4_threads/512 | iteration | 13261 | 47750.456376018024 | 47622.563305934695 | ns |
| sgemv_v1_4_threads/1024 | sgemv_v1_4_threads/1024 | iteration | 5073 | 137631.56869689125 | 137276.145278927 | ns |
| sgemv_v1_4_threads/2048 | sgemv_v1_4_threads/2048 | iteration | 1888 | 368909.69809295924 | 367984.2817796581 | ns |
