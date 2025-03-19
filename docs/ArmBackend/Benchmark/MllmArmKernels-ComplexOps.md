# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-16T14:42:57+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_baseline/64 | softmax_baseline/64 | iteration | 3612079 | 194.7891654738771 | 193.5553416744208 | ns |
| softmax_baseline/128 | softmax_baseline/128 | iteration | 1828927 | 385.36723553265807 | 382.78917037148005 | ns |
| softmax_baseline/256 | softmax_baseline/256 | iteration | 936749 | 745.9334304000212 | 740.8548618680142 | ns |
| softmax_baseline/512 | softmax_baseline/512 | iteration | 484343 | 1454.1385402262076 | 1445.4654036498923 | ns |
| softmax_baseline/1024 | softmax_baseline/1024 | iteration | 243805 | 2891.7310516170837 | 2872.2700846988387 | ns |
| softmax_baseline/2048 | softmax_baseline/2048 | iteration | 121898 | 5753.3838863564515 | 5713.45209929613 | ns |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 9317616 | 75.71922432017817 | 75.20797240409996 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4731909 | 149.75834572464086 | 148.76431731886643 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2458539 | 287.35197691433734 | 285.56163965672255 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1217759 | 558.2621437982406 | 554.5014777143921 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 642247 | 1094.978271608207 | 1087.7032714827785 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 324385 | 2170.2543859521743 | 2155.4863757571998 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 144785 | 4882.917622830576 | 4848.763159167044 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 36797 | 19094.94575598094 | 18968.18335733888 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 9476 | 74407.13707966209 | 73857.92623469808 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2318 | 302781.7622816304 | 300052.6945642796 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 584 | 1241537.9742503306 | 1229206.9760273953 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 126 | 5261313.238103564 | 5207570.682539684 | ns |
| softmax_v1_f32_kxk_4_threads/64 | softmax_v1_f32_kxk_4_threads/64 | iteration | 105357 | 6639.286046695041 | 6612.655324278408 | ns |
| softmax_v1_f32_kxk_4_threads/128 | softmax_v1_f32_kxk_4_threads/128 | iteration | 51135 | 13928.342250845011 | 13861.732433753827 | ns |
| softmax_v1_f32_kxk_4_threads/256 | softmax_v1_f32_kxk_4_threads/256 | iteration | 17703 | 39722.41636986955 | 39531.83821951087 | ns |
| softmax_v1_f32_kxk_4_threads/512 | softmax_v1_f32_kxk_4_threads/512 | iteration | 5068 | 138139.64443519452 | 137509.47513812137 | ns |
| softmax_v1_f32_kxk_4_threads/1024 | softmax_v1_f32_kxk_4_threads/1024 | iteration | 1305 | 540152.0996525799 | 535278.3501915723 | ns |
| softmax_v1_f32_kxk_4_threads/2048 | softmax_v1_f32_kxk_4_threads/2048 | iteration | 290 | 2426473.7758946056 | 2409975.5241379333 | ns |
| softmax_v1_f16/64 | softmax_v1_f16/64 | iteration | 7654803 | 89.25950909831731 | 88.63872447141999 | ns |
| softmax_v1_f16/128 | softmax_v1_f16/128 | iteration | 4104060 | 171.71564304289222 | 170.69429516137714 | ns |
| softmax_v1_f16/256 | softmax_v1_f16/256 | iteration | 2057513 | 342.2983383320571 | 340.24103906026295 | ns |
| softmax_v1_f16/512 | softmax_v1_f16/512 | iteration | 1047184 | 674.5517415898277 | 670.3789505951197 | ns |
| softmax_v1_f16/1024 | softmax_v1_f16/1024 | iteration | 530881 | 1330.826237932376 | 1321.7435413962803 | ns |
| softmax_v1_f16/2048 | softmax_v1_f16/2048 | iteration | 262230 | 2639.9258668052694 | 2621.8936124775832 | ns |
| softmax_v1_f16_kxk/64 | softmax_v1_f16_kxk/64 | iteration | 123272 | 5716.639747660466 | 5679.131587059517 | ns |
| softmax_v1_f16_kxk/128 | softmax_v1_f16_kxk/128 | iteration | 31889 | 22039.262818702296 | 21883.00664806054 | ns |
| softmax_v1_f16_kxk/256 | softmax_v1_f16_kxk/256 | iteration | 8046 | 87962.0372833628 | 87300.23663932367 | ns |
| softmax_v1_f16_kxk/512 | softmax_v1_f16_kxk/512 | iteration | 2034 | 358317.9695225864 | 354573.884955751 | ns |
| softmax_v1_f16_kxk/1024 | softmax_v1_f16_kxk/1024 | iteration | 486 | 1385728.7366265133 | 1372459.7078189265 | ns |
| softmax_v1_f16_kxk/2048 | softmax_v1_f16_kxk/2048 | iteration | 120 | 5702204.858243931 | 5647163.700000017 | ns |
| softmax_v1_f16_kxk_4_threads/64 | softmax_v1_f16_kxk_4_threads/64 | iteration | 92569 | 7017.550195208711 | 6985.054132592971 | ns |
| softmax_v1_f16_kxk_4_threads/128 | softmax_v1_f16_kxk_4_threads/128 | iteration | 47916 | 14760.816219863447 | 14695.802112029449 | ns |
| softmax_v1_f16_kxk_4_threads/256 | softmax_v1_f16_kxk_4_threads/256 | iteration | 15949 | 44186.57382651678 | 43998.83240328529 | ns |
| softmax_v1_f16_kxk_4_threads/512 | softmax_v1_f16_kxk_4_threads/512 | iteration | 4402 | 159200.97865621268 | 158697.25374829673 | ns |
| softmax_v1_f16_kxk_4_threads/1024 | softmax_v1_f16_kxk_4_threads/1024 | iteration | 1147 | 611549.830837573 | 608472.6599825653 | ns |
| softmax_v1_f16_kxk_4_threads/2048 | softmax_v1_f16_kxk_4_threads/2048 | iteration | 286 | 2481735.321828262 | 2464071.2377622174 | ns |
| transpose_fp32_bshd2bhsd | transpose_fp32_bshd2bhsd | iteration | 190 | 3616522.4789568274 | 3584213.315789504 | ns |
