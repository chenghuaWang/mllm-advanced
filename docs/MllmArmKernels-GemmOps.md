# MllmArmKernels-GemmOps Benchmark Results

device: xiaomi 12s

data: 2025-02-16T14:43:53+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemmOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| sgemm_v1/64 | sgemm_v1/64 | iteration | 11860 | 59449.62057710624 | 58866.5992411467 | ns |
| sgemm_v1/128 | sgemm_v1/128 | iteration | 1535 | 473169.00780576153 | 468209.3074918567 | ns |
| sgemm_v1/256 | sgemm_v1/256 | iteration | 203 | 3951685.394165037 | 3913716.97044335 | ns |
| sgemm_v1/512 | sgemm_v1/512 | iteration | 24 | 36923465.6682238 | 36518618.37499999 | ns |
| sgemm_v1/1024 | sgemm_v1/1024 | iteration | 3 | 243860641.99885973 | 241500610.3333335 | ns |
| sgemm_v1/2048 | sgemm_v1/2048 | iteration | 1 | 1611009947.0242858 | 1594500937.0000002 | ns |
| sgemm_v1_4_threads/64 | sgemm_v1_4_threads/64 | iteration | 12554 | 72612.16536575864 | 54422.24231320695 | ns |
| sgemm_v1_4_threads/128 | sgemm_v1_4_threads/128 | iteration | 1777 | 763580.0624669823 | 379159.5278559367 | ns |
| sgemm_v1_4_threads/256 | sgemm_v1_4_threads/256 | iteration | 236 | 5459756.796619208 | 2711461.6228813534 | ns |
| sgemm_v1_4_threads/512 | sgemm_v1_4_threads/512 | iteration | 46 | 32052186.348111086 | 15942174.086956501 | ns |
| sgemm_v1_4_threads/1024 | sgemm_v1_4_threads/1024 | iteration | 4 | 310969804.50092816 | 153983746.0000002 | ns |
| sgemm_v1_4_threads/2048 | sgemm_v1_4_threads/2048 | iteration | 1 | 1252109998.9535286 | 623400977.9999994 | ns |
| sgemm_v1_128x1536x8960_4_threads | sgemm_v1_128x1536x8960_4_threads | iteration | 4 | 400685689.994134 | 198702697.50000036 | ns |
| sgemm_mk_kn_v1/64 | sgemm_mk_kn_v1/64 | iteration | 32318 | 22340.855560534004 | 22053.55096231201 | ns |
| sgemm_mk_kn_v1/128 | sgemm_mk_kn_v1/128 | iteration | 5572 | 127458.2173314102 | 125823.17300789682 | ns |
| sgemm_mk_kn_v1/256 | sgemm_mk_kn_v1/256 | iteration | 581 | 1203965.683321015 | 1190564.2891566264 | ns |
| sgemm_mk_kn_v1/512 | sgemm_mk_kn_v1/512 | iteration | 65 | 10903792.461165443 | 10779544.230769252 | ns |
| sgemm_mk_kn_v1/1024 | sgemm_mk_kn_v1/1024 | iteration | 9 | 77485798.55413279 | 76336024.88888869 | ns |
| sgemm_mk_kn_v1/2048 | sgemm_mk_kn_v1/2048 | iteration | 2 | 468567030.999111 | 462137138.4999993 | ns |
| sgemm_mk_kn_v1_4_threads/64 | sgemm_mk_kn_v1_4_threads/64 | iteration | 26987 | 25885.943529485034 | 25686.02719828076 | ns |
| sgemm_mk_kn_v1_4_threads/128 | sgemm_mk_kn_v1_4_threads/128 | iteration | 5452 | 120655.77035908526 | 119747.08235509876 | ns |
| sgemm_mk_kn_v1_4_threads/256 | sgemm_mk_kn_v1_4_threads/256 | iteration | 701 | 992560.7018094355 | 984558.592011414 | ns |
| sgemm_mk_kn_v1_4_threads/512 | sgemm_mk_kn_v1_4_threads/512 | iteration | 86 | 11628532.58185435 | 5910574.75581396 | ns |
| sgemm_mk_kn_v1_4_threads/1024 | sgemm_mk_kn_v1_4_threads/1024 | iteration | 18 | 77611927.0562453 | 38475583.1666666 | ns |
| sgemm_mk_kn_v1_4_threads/2048 | sgemm_mk_kn_v1_4_threads/2048 | iteration | 5 | 294570812.4083467 | 145658225.99999976 | ns |
| sgemm_mk_kn_v1_128x1536x8960_4_threads | sgemm_mk_kn_v1_128x1536x8960_4_threads | iteration | 8 | 161946275.99907106 | 80268842.00000018 | ns |
