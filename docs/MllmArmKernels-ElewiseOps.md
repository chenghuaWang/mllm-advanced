# MllmArmKernels-ElewiseOps Benchmark Results

device: <your device name>

data: 2025-02-03T21:25:00+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 588262 | 1191.6265354511704 | 1201.559543196827 | ns |
| add_f32/128 | add_f32/128 | iteration | 40469 | 17498.07484737269 | 17418.401023003582 | ns |
| add_f32/256 | add_f32/256 | iteration | 10372 | 68154.74257621261 | 67853.43116081739 | ns |
| add_f32/512 | add_f32/512 | iteration | 2694 | 261808.55902005747 | 260484.61469933653 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 596 | 1185758.8070470614 | 1177879.2164429391 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 132 | 5226478.878788065 | 5178726.3939393675 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 22414 | 31377.07450700831 | 31255.448425092094 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 6159 | 113410.91686957405 | 112941.72576717495 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 1840 | 377760.658152041 | 376090.32228259573 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 474 | 1486174.1666667652 | 1481010.7004219438 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 116 | 5919166.620690111 | 5884179.86206904 | ns |
| add_f16/64 | add_f16/64 | iteration | 826018 | 802.5723906763642 | 805.4174340026308 | ns |
| add_f16/128 | add_f16/128 | iteration | 345127 | 2003.2680230638853 | 2028.6990817861238 | ns |
| add_f16/256 | add_f16/256 | iteration | 18596 | 37559.219617119554 | 37467.43009248716 | ns |
| add_f16/512 | add_f16/512 | iteration | 4520 | 149794.2654866329 | 149393.2190265343 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 1122 | 626592.6675580807 | 624453.5650623057 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 262 | 2649965.000000705 | 2631840.251908495 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 152238 | 4591.006056333043 | 4597.18725285667 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 12248 | 57063.0589482451 | 56883.30502941387 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 3451 | 203436.6618371023 | 202803.6096783609 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 934 | 764654.9014988703 | 762258.8608137054 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 229 | 3071239.0567678716 | 3058274.8209606847 | ns |
