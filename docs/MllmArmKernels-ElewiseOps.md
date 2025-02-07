# MllmArmKernels-ElewiseOps Benchmark Results

device: xiaomi 12s

data: 2025-02-07T20:09:30+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-ElewiseOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| add_f32/64 | add_f32/64 | iteration | 601572 | 1154.4962498857033 | 1164.3592271580935 | ns |
| add_f32/128 | add_f32/128 | iteration | 38925 | 18173.03401413245 | 18045.741836868354 | ns |
| add_f32/256 | add_f32/256 | iteration | 9995 | 70827.48057730934 | 70276.6832416186 | ns |
| add_f32/512 | add_f32/512 | iteration | 2374 | 297147.1116181072 | 294391.46082559705 | ns |
| add_f32/1024 | add_f32/1024 | iteration | 553 | 1317566.01689167 | 1300841.1717902631 | ns |
| add_f32/2048 | add_f32/2048 | iteration | 111 | 6422124.1795625705 | 6334308.324324326 | ns |
| add_f32_4_threads/128 | add_f32_4_threads/128 | iteration | 23915 | 29189.83341177457 | 29051.168388039965 | ns |
| add_f32_4_threads/256 | add_f32_4_threads/256 | iteration | 8930 | 78433.84258653749 | 78021.52922730561 | ns |
| add_f32_4_threads/512 | add_f32_4_threads/512 | iteration | 3027 | 232698.85715408245 | 231525.57284440246 | ns |
| add_f32_4_threads/1024 | add_f32_4_threads/1024 | iteration | 807 | 887159.3170922222 | 881730.677819104 | ns |
| add_f32_4_threads/2048 | add_f32_4_threads/2048 | iteration | 199 | 3558625.990186172 | 3526925.206030045 | ns |
| add_f16/64 | add_f16/64 | iteration | 841006 | 793.5043106093453 | 793.330393600763 | ns |
| add_f16/128 | add_f16/128 | iteration | 334022 | 2080.4952507324497 | 2097.7244163555456 | ns |
| add_f16/256 | add_f16/256 | iteration | 16642 | 42045.0489387202 | 41773.523314484635 | ns |
| add_f16/512 | add_f16/512 | iteration | 3796 | 164449.01515314728 | 163046.94388830927 | ns |
| add_f16/1024 | add_f16/1024 | iteration | 1067 | 725448.1010254421 | 717820.7675726212 | ns |
| add_f16/2048 | add_f16/2048 | iteration | 195 | 3642409.4824514426 | 3594107.5794873713 | ns |
| add_f16_4_threads/128 | add_f16_4_threads/128 | iteration | 83318 | 8124.625431163123 | 8126.767253172428 | ns |
| add_f16_4_threads/256 | add_f16_4_threads/256 | iteration | 15219 | 46373.65239823268 | 46157.83822852533 | ns |
| add_f16_4_threads/512 | add_f16_4_threads/512 | iteration | 5507 | 128238.01643070273 | 127561.18830581514 | ns |
| add_f16_4_threads/1024 | add_f16_4_threads/1024 | iteration | 1572 | 442018.508132809 | 439447.7003816123 | ns |
| add_f16_4_threads/2048 | add_f16_4_threads/2048 | iteration | 404 | 1764412.6735219043 | 1750940.2351486625 | ns |
