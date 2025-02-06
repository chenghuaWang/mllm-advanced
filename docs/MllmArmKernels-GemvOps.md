# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-06T17:09:14+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2785316 | 252.62102181396017 | 251.41157233146976 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 286291 | 2474.461264925776 | 2462.8719729226564 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 75404 | 9443.684870795612 | 9395.333430587236 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 23601 | 27671.066649497825 | 27533.338460234736 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7784 | 91354.50115657094 | 90917.05999486125 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1615 | 435127.4185794501 | 433045.5027863778 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1679081 | 419.00096660638354 | 416.8741496092207 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 261826 | 2695.871972201784 | 2681.899979375617 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 69024 | 10186.697597946451 | 10139.349747913786 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19297 | 36230.27273738594 | 36051.73990775771 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 5789 | 118786.99550970952 | 118216.52254275348 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1325 | 526058.7660463983 | 523420.16981132055 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 527260 | 1334.3852520530945 | 1327.7688009710578 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 151388 | 4640.942961120401 | 4615.470063677436 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49838 | 14128.717665081202 | 14057.960371603986 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 14813 | 47864.37237604604 | 47630.80820900557 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3183 | 220952.48853349476 | 219783.89161168702 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 950 | 729285.0873598485 | 725409.9336842083 | ns |
