# MllmArmKernels-ComplexOps Benchmark Results

device: xiaomi 12s

data: 2025-02-14T15:11:28+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ComplexOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_baseline/64 | softmax_baseline/64 | iteration | 3621781 | 194.89267655656627 | 193.52294106131765 | ns |
| softmax_baseline/128 | softmax_baseline/128 | iteration | 1842096 | 382.73815046061026 | 380.250933718981 | ns |
| softmax_baseline/256 | softmax_baseline/256 | iteration | 947935 | 742.2527314856493 | 737.4231017949543 | ns |
| softmax_baseline/512 | softmax_baseline/512 | iteration | 480422 | 1455.1408282727016 | 1445.442217467143 | ns |
| softmax_baseline/1024 | softmax_baseline/1024 | iteration | 244053 | 2883.2161539412764 | 2865.086710673502 | ns |
| softmax_baseline/2048 | softmax_baseline/2048 | iteration | 122559 | 5743.709829366755 | 5708.305477361928 | ns |
| softmax_v1_f32/64 | softmax_v1_f32/64 | iteration | 9306383 | 75.62159670393338 | 75.15925714641229 | ns |
| softmax_v1_f32/128 | softmax_v1_f32/128 | iteration | 4724021 | 148.92306299765855 | 147.9579169948653 | ns |
| softmax_v1_f32/256 | softmax_v1_f32/256 | iteration | 2455048 | 286.7166230635306 | 284.8400727806541 | ns |
| softmax_v1_f32/512 | softmax_v1_f32/512 | iteration | 1264684 | 556.7675814794621 | 553.2257306963636 | ns |
| softmax_v1_f32/1024 | softmax_v1_f32/1024 | iteration | 644339 | 1093.9260513598997 | 1086.738623612726 | ns |
| softmax_v1_f32/2048 | softmax_v1_f32/2048 | iteration | 324795 | 2169.9446943063544 | 2155.641604704505 | ns |
| softmax_v1_f32_kxk/64 | softmax_v1_f32_kxk/64 | iteration | 145130 | 4856.055053964569 | 4825.759353682908 | ns |
| softmax_v1_f32_kxk/128 | softmax_v1_f32_kxk/128 | iteration | 36942 | 19093.746900374234 | 18965.344648367718 | ns |
| softmax_v1_f32_kxk/256 | softmax_v1_f32_kxk/256 | iteration | 9442 | 74344.3688834522 | 73838.77568311802 | ns |
| softmax_v1_f32_kxk/512 | softmax_v1_f32_kxk/512 | iteration | 2279 | 306284.94295302226 | 303620.44361562113 | ns |
| softmax_v1_f32_kxk/1024 | softmax_v1_f32_kxk/1024 | iteration | 590 | 1229834.3034255933 | 1217987.6406779664 | ns |
| softmax_v1_f32_kxk/2048 | softmax_v1_f32_kxk/2048 | iteration | 126 | 5221979.992162984 | 5162931.079365066 | ns |
| softmax_v1_f32_kxk_4_threads/64 | softmax_v1_f32_kxk_4_threads/64 | iteration | 106019 | 6665.278846238522 | 6637.113809788802 | ns |
| softmax_v1_f32_kxk_4_threads/128 | softmax_v1_f32_kxk_4_threads/128 | iteration | 49865 | 13890.47197397015 | 13836.583856412313 | ns |
| softmax_v1_f32_kxk_4_threads/256 | softmax_v1_f32_kxk_4_threads/256 | iteration | 17774 | 39934.72144752469 | 39773.885844492004 | ns |
| softmax_v1_f32_kxk_4_threads/512 | softmax_v1_f32_kxk_4_threads/512 | iteration | 5035 | 139672.11797667894 | 139064.22661370342 | ns |
| softmax_v1_f32_kxk_4_threads/1024 | softmax_v1_f32_kxk_4_threads/1024 | iteration | 1287 | 537898.9409528696 | 533576.7668997676 | ns |
| softmax_v1_f32_kxk_4_threads/2048 | softmax_v1_f32_kxk_4_threads/2048 | iteration | 291 | 2430350.6220302056 | 2410528.542955325 | ns |
| softmax_v1_f16/64 | softmax_v1_f16/64 | iteration | 7433834 | 89.89576684114009 | 89.36416027046079 | ns |
| softmax_v1_f16/128 | softmax_v1_f16/128 | iteration | 4103590 | 172.07175351165432 | 170.90583976469324 | ns |
| softmax_v1_f16/256 | softmax_v1_f16/256 | iteration | 2053899 | 342.2805537106506 | 340.15423591909695 | ns |
| softmax_v1_f16/512 | softmax_v1_f16/512 | iteration | 1038857 | 673.5195700723946 | 669.0205033031477 | ns |
| softmax_v1_f16/1024 | softmax_v1_f16/1024 | iteration | 531186 | 1328.9411769496921 | 1320.317679682823 | ns |
| softmax_v1_f16/2048 | softmax_v1_f16/2048 | iteration | 265967 | 2636.4235638293385 | 2620.309023299885 | ns |
| softmax_v1_f16_kxk/64 | softmax_v1_f16_kxk/64 | iteration | 123157 | 5715.274763265637 | 5677.143353605538 | ns |
| softmax_v1_f16_kxk/128 | softmax_v1_f16_kxk/128 | iteration | 31936 | 22035.428263003625 | 21896.341714679358 | ns |
| softmax_v1_f16_kxk/256 | softmax_v1_f16_kxk/256 | iteration | 8036 | 87602.2093080892 | 87036.34992533585 | ns |
| softmax_v1_f16_kxk/512 | softmax_v1_f16_kxk/512 | iteration | 2036 | 346464.984283498 | 343790.72789783863 | ns |
| softmax_v1_f16_kxk/1024 | softmax_v1_f16_kxk/1024 | iteration | 508 | 1385918.1220420476 | 1374008.2499999958 | ns |
| softmax_v1_f16_kxk/2048 | softmax_v1_f16_kxk/2048 | iteration | 120 | 5864319.874914751 | 5800024.716666652 | ns |
| softmax_v1_f16_kxk_4_threads/64 | softmax_v1_f16_kxk_4_threads/64 | iteration | 94312 | 7075.489248402665 | 7044.215624734935 | ns |
| softmax_v1_f16_kxk_4_threads/128 | softmax_v1_f16_kxk_4_threads/128 | iteration | 47613 | 14771.888055475312 | 14715.530695398389 | ns |
| softmax_v1_f16_kxk_4_threads/256 | softmax_v1_f16_kxk_4_threads/256 | iteration | 15602 | 44299.1883736401 | 44120.34963466233 | ns |
| softmax_v1_f16_kxk_4_threads/512 | softmax_v1_f16_kxk_4_threads/512 | iteration | 4455 | 157997.97732721677 | 157462.80875420844 | ns |
| softmax_v1_f16_kxk_4_threads/1024 | softmax_v1_f16_kxk_4_threads/1024 | iteration | 1142 | 612337.6383449011 | 609405.4299474617 | ns |
| softmax_v1_f16_kxk_4_threads/2048 | softmax_v1_f16_kxk_4_threads/2048 | iteration | 284 | 2462021.345014415 | 2445546.8873239425 | ns |
| transpose_fp32_bshd2bhsd | transpose_fp32_bshd2bhsd | iteration | 219 | 3088719.557043692 | 3061966.566210065 | ns |
