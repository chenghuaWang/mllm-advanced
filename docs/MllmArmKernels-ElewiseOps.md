# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-16T14:41:27+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 552535 | 1245.9289476018769 | 1259.4080936050457 | ns |
| add_f32/128 | add_f32/128 | iteration | 36025 | 19475.4781822579 | 19400.124996529124 | ns |
| add_f32/256 | add_f32/256 | iteration | 9417 | 74757.40152778172 | 74347.06105978455 | ns |
| add_f32/512 | add_f32/512 | iteration | 2164 | 329837.18449297204 | 326548.60720887664 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 508 | 1426073.3361707956 | 1411041.0964567175 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 105 | 6672329.361373115 | 6599923.5333333295 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 18424 | 38486.54868450121 | 38200.15056447656 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 8112 | 84534.00938052278 | 83892.35182446399 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 3002 | 234511.76346208324 | 233154.3041305375 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 805 | 866497.6628294782 | 861258.8633540568 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 199 | 3494413.431509206 | 3463000.648241191 | ns |
| add_f16/64 | add_f16/64 | iteration | 825307 | 815.0744536345472 | 815.0767156941436 | ns |
| add_f16/128 | add_f16/128 | iteration | 330081 | 2075.3627700609773 | 2105.638367552698 | ns |
| add_f16/256 | add_f16/256 | iteration | 16569 | 42872.981599197796 | 42673.806204346925 | ns |
| add_f16/512 | add_f16/512 | iteration | 4164 | 176929.25309009515 | 175554.71565805486 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 949 | 737776.3070118024 | 729781.4383561804 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 215 | 3266999.4828486163 | 3225589.190697742 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 81041 | 8248.899135368983 | 8284.284337551033 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 15046 | 46301.71461649824 | 46099.59052238553 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 5468 | 129782.8977012594 | 129157.27907824893 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 1555 | 448696.8217513425 | 446053.7974276436 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 406 | 1723863.5431603517 | 1709849.0197043929 | ns |
