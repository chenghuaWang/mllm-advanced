# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T15:06:30+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2789125 | 252.5917999367088 | 250.8705127952315 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 266334 | 2625.1713862839047 | 2610.091663099717 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 74033 | 9374.946537349797 | 9319.081071954404 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 24454 | 27751.002698704513 | 27586.56109429949 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7873 | 90311.70608344677 | 89764.47694652608 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1613 | 435849.7365158148 | 433386.80595164263 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1676379 | 419.77483552430635 | 417.16185480729604 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 269601 | 2610.824459071786 | 2593.790245585141 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 68881 | 10070.639639305522 | 10004.853733250091 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19434 | 35738.34033129272 | 35515.59663476383 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 5545 | 120959.5262408265 | 120160.22218214608 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1320 | 525883.3643893013 | 522358.4295454543 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 555861 | 1265.0201471302762 | 1257.1469486076578 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 149869 | 4603.321620840108 | 4576.036071502438 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 48689 | 14417.791780453714 | 14335.410811476902 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 13726 | 50820.01457119653 | 50495.74843362973 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3172 | 221884.67118663163 | 220605.7547288774 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 876 | 791746.3378972783 | 786941.8367579893 | ns |
