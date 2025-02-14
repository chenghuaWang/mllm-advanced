# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-14T15:10:38+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2797159 | 251.64580669318016 | 250.10786623141553 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 263468 | 2699.715312677468 | 2683.4654986563824 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 75290 | 9362.355146916683 | 9302.219723734892 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 23741 | 28519.24590429379 | 28332.18348005561 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7902 | 90588.25423947735 | 89998.72070361934 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1609 | 437100.10129706026 | 434209.74829086434 | ns |
| hgemv_v1_4_threads/64 | hgemv_v1_4_threads/64 | iteration | 100109 | 6832.1425445304485 | 6803.474892367323 | ns |
| hgemv_v1_4_threads/128 | hgemv_v1_4_threads/128 | iteration | 60043 | 11829.813616866806 | 11788.268557533767 | ns |
| hgemv_v1_4_threads/256 | hgemv_v1_4_threads/256 | iteration | 35099 | 20012.133820859894 | 19938.465682782964 | ns |
| hgemv_v1_4_threads/512 | hgemv_v1_4_threads/512 | iteration | 18361 | 35319.483796393586 | 35199.531833778114 | ns |
| hgemv_v1_4_threads/1024 | hgemv_v1_4_threads/1024 | iteration | 7073 | 99238.3299870518 | 98908.62773929033 | ns |
| hgemv_v1_4_threads/2048 | hgemv_v1_4_threads/2048 | iteration | 2288 | 308414.57691758795 | 307385.06687062944 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1593835 | 432.726691897866 | 429.8717414286933 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 267870 | 2647.1035800908376 | 2629.940575652367 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 68749 | 10106.383365531658 | 10037.747487236169 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19752 | 35849.28635059937 | 35611.31723369786 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 5910 | 116215.0133694262 | 115498.26142131942 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1326 | 527364.2926011585 | 523717.3205128193 | ns |
| hgemv_v2_hp_4_threads/64 | hgemv_v2_hp_4_threads/64 | iteration | 99744 | 6985.499017512362 | 6959.455325633627 | ns |
| hgemv_v2_hp_4_threads/128 | hgemv_v2_hp_4_threads/128 | iteration | 108656 | 6358.422737643462 | 6332.094444853474 | ns |
| hgemv_v2_hp_4_threads/256 | hgemv_v2_hp_4_threads/256 | iteration | 67867 | 9997.03770623476 | 9957.203928271501 | ns |
| hgemv_v2_hp_4_threads/512 | hgemv_v2_hp_4_threads/512 | iteration | 30126 | 23697.831939139454 | 23606.72661488418 | ns |
| hgemv_v2_hp_4_threads/1024 | hgemv_v2_hp_4_threads/1024 | iteration | 7154 | 95957.54990173358 | 95627.65026558563 | ns |
| hgemv_v2_hp_4_threads/2048 | hgemv_v2_hp_4_threads/2048 | iteration | 1779 | 397936.51602966496 | 396400.7745924677 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 563797 | 1181.423954039742 | 1173.7981046369573 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 147241 | 4619.03761850103 | 4587.308589319533 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49714 | 14227.949491412659 | 14137.68441485297 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 15315 | 46715.06013662241 | 46385.51446294479 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3168 | 221909.656250897 | 220587.042613637 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 942 | 735427.0615488896 | 730939.4012738869 | ns |
| sgemv_v1_4_threads/64 | sgemv_v1_4_threads/64 | iteration | 90280 | 7528.188912161119 | 7496.270968099258 | ns |
| sgemv_v1_4_threads/128 | sgemv_v1_4_threads/128 | iteration | 61205 | 11477.236679806354 | 11435.818593252168 | ns |
| sgemv_v1_4_threads/256 | sgemv_v1_4_threads/256 | iteration | 33365 | 20688.65203102399 | 20612.7430840702 | ns |
| sgemv_v1_4_threads/512 | sgemv_v1_4_threads/512 | iteration | 15472 | 45943.18226450583 | 45774.487138055825 | ns |
| sgemv_v1_4_threads/1024 | sgemv_v1_4_threads/1024 | iteration | 4449 | 157944.08383641485 | 157356.37086985828 | ns |
| sgemv_v1_4_threads/2048 | sgemv_v1_4_threads/2048 | iteration | 1956 | 360492.60787503 | 359212.5168711629 | ns |
