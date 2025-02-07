# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T20:10:12+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2759381 | 254.93034416301757 | 253.45554057232405 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 258166 | 2732.3678524607776 | 2716.3485664262535 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 75321 | 9278.092537379342 | 9224.391789806294 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 24801 | 29854.802548318537 | 29687.592234184114 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7789 | 91730.6658119862 | 91221.96867377065 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1616 | 435620.9387383793 | 433256.36571782193 | ns |
| hgemv_v1_4_threads/64 | hgemv_v1_4_threads/64 | iteration | 96583 | 6984.9394405966705 | 6957.090533530751 | ns |
| hgemv_v1_4_threads/128 | hgemv_v1_4_threads/128 | iteration | 61047 | 11949.163316764216 | 11906.790882434841 | ns |
| hgemv_v1_4_threads/256 | hgemv_v1_4_threads/256 | iteration | 35576 | 20242.131268512003 | 20173.278136946254 | ns |
| hgemv_v1_4_threads/512 | hgemv_v1_4_threads/512 | iteration | 18748 | 35228.39657571001 | 35095.642948581124 | ns |
| hgemv_v1_4_threads/1024 | hgemv_v1_4_threads/1024 | iteration | 7641 | 92268.47755480577 | 91908.01766784459 | ns |
| hgemv_v1_4_threads/2048 | hgemv_v1_4_threads/2048 | iteration | 2123 | 333163.9580821221 | 331938.31606217567 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1564942 | 440.89410214412396 | 438.29988843036966 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 260843 | 2689.719804601521 | 2673.842376448667 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 70739 | 10072.824198692186 | 10019.207905115985 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19517 | 35889.507608570326 | 35681.72178101142 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 5917 | 121440.57901038018 | 120703.27936454276 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1320 | 527167.1393998214 | 524388.7598484866 | ns |
| hgemv_v2_hp_4_threads/64 | hgemv_v2_hp_4_threads/64 | iteration | 48152 | 15076.318637016862 | 14995.905964445936 | ns |
| hgemv_v2_hp_4_threads/128 | hgemv_v2_hp_4_threads/128 | iteration | 39717 | 16361.435531364748 | 16270.626658609666 | ns |
| hgemv_v2_hp_4_threads/256 | hgemv_v2_hp_4_threads/256 | iteration | 68824 | 10328.155875885337 | 10283.743112867616 | ns |
| hgemv_v2_hp_4_threads/512 | hgemv_v2_hp_4_threads/512 | iteration | 28945 | 24320.572050752293 | 24218.623803765764 | ns |
| hgemv_v2_hp_4_threads/1024 | hgemv_v2_hp_4_threads/1024 | iteration | 6639 | 106331.98870403816 | 105949.58999849424 | ns |
| hgemv_v2_hp_4_threads/2048 | hgemv_v2_hp_4_threads/2048 | iteration | 1684 | 419044.09798897745 | 417518.7226840863 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 535030 | 1245.001641027085 | 1238.0304319383918 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 151014 | 4630.062583630782 | 4605.609122333019 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49659 | 14034.778650364282 | 13950.654926599453 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 14677 | 47387.24207945134 | 47128.13797097495 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3179 | 221361.8015118934 | 219987.99182132675 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 930 | 741399.6408583336 | 736839.4430107517 | ns |
| sgemv_v1_4_threads/64 | sgemv_v1_4_threads/64 | iteration | 97012 | 7284.711973754237 | 7255.557230033399 | ns |
| sgemv_v1_4_threads/128 | sgemv_v1_4_threads/128 | iteration | 60350 | 11189.342535262142 | 11149.997415078717 | ns |
| sgemv_v1_4_threads/256 | sgemv_v1_4_threads/256 | iteration | 33335 | 20967.87814646487 | 20896.65525723716 | ns |
| sgemv_v1_4_threads/512 | sgemv_v1_4_threads/512 | iteration | 14659 | 53016.49505496301 | 52826.63032949042 | ns |
| sgemv_v1_4_threads/1024 | sgemv_v1_4_threads/1024 | iteration | 5005 | 138835.91408712917 | 138380.7988011989 | ns |
| sgemv_v1_4_threads/2048 | sgemv_v1_4_threads/2048 | iteration | 1872 | 376090.077458148 | 374399.96153846383 | ns |
