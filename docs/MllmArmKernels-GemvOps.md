# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-15T11:05:47+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2787511 | 251.70349103428816 | 251.04370350466775 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 278597 | 2557.013729510003 | 2551.698245135445 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 73844 | 9452.182696067648 | 9430.790314717517 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 24789 | 29552.138770926973 | 29487.952882326837 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7957 | 89032.35088713374 | 88837.24104562019 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1611 | 434725.29298707977 | 433822.3600248297 | ns |
| hgemv_v1_4_threads/64 | hgemv_v1_4_threads/64 | iteration | 94082 | 7214.4811122632345 | 7194.5727344231655 | ns |
| hgemv_v1_4_threads/128 | hgemv_v1_4_threads/128 | iteration | 56031 | 12333.759240283121 | 12299.505541575192 | ns |
| hgemv_v1_4_threads/256 | hgemv_v1_4_threads/256 | iteration | 34651 | 20560.973161556652 | 20505.46746125652 | ns |
| hgemv_v1_4_threads/512 | hgemv_v1_4_threads/512 | iteration | 17941 | 35799.73942337213 | 35703.637924307484 | ns |
| hgemv_v1_4_threads/1024 | hgemv_v1_4_threads/1024 | iteration | 8067 | 87325.16214324135 | 87095.08404611377 | ns |
| hgemv_v1_4_threads/2048 | hgemv_v1_4_threads/2048 | iteration | 2283 | 306882.2317120737 | 306025.1992991677 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1580062 | 434.77101278134245 | 433.6419634166252 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 266951 | 2634.9635176489564 | 2629.374169791461 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 68710 | 10184.552074235035 | 10162.275593072329 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19710 | 35734.97483435449 | 35659.82831050236 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 6202 | 122963.9043892223 | 122708.4072879714 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1327 | 526088.0587746395 | 525027.7498116033 | ns |
| hgemv_v2_hp_4_threads/64 | hgemv_v2_hp_4_threads/64 | iteration | 111378 | 5946.83055902166 | 5929.900132880847 | ns |
| hgemv_v2_hp_4_threads/128 | hgemv_v2_hp_4_threads/128 | iteration | 113131 | 6190.1279400820795 | 6171.235788599045 | ns |
| hgemv_v2_hp_4_threads/256 | hgemv_v2_hp_4_threads/256 | iteration | 68201 | 9821.78520858732 | 9790.166727760612 | ns |
| hgemv_v2_hp_4_threads/512 | hgemv_v2_hp_4_threads/512 | iteration | 30488 | 22900.587936870757 | 22831.072946733148 | ns |
| hgemv_v2_hp_4_threads/1024 | hgemv_v2_hp_4_threads/1024 | iteration | 7567 | 91902.5180401908 | 91657.84313466385 | ns |
| hgemv_v2_hp_4_threads/2048 | hgemv_v2_hp_4_threads/2048 | iteration | 1816 | 385930.18666988786 | 384913.70099118917 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 585108 | 1135.7864872797802 | 1133.1192514886168 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 153582 | 4598.6980179375105 | 4589.26598820174 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49506 | 14157.846179997994 | 14131.13535733041 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 14976 | 46426.4003059927 | 46332.698651175204 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3191 | 220204.95456340746 | 219753.54152303285 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 952 | 732341.725832906 | 730748.4737394968 | ns |
| sgemv_v1_4_threads/64 | sgemv_v1_4_threads/64 | iteration | 88812 | 7257.747286317538 | 7236.8247308922 | ns |
| sgemv_v1_4_threads/128 | sgemv_v1_4_threads/128 | iteration | 62511 | 11159.182639999772 | 11125.945177648739 | ns |
| sgemv_v1_4_threads/256 | sgemv_v1_4_threads/256 | iteration | 33855 | 20644.351824167075 | 20591.78419731211 | ns |
| sgemv_v1_4_threads/512 | sgemv_v1_4_threads/512 | iteration | 15532 | 44640.72154404883 | 44525.35462271453 | ns |
| sgemv_v1_4_threads/1024 | sgemv_v1_4_threads/1024 | iteration | 4478 | 154244.94283226397 | 153771.99642697687 | ns |
| sgemv_v1_4_threads/2048 | sgemv_v1_4_threads/2048 | iteration | 1979 | 426213.7074402791 | 403178.1844365823 | ns |
