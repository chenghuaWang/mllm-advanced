# MllmArmKernels-GemvOps Benchmark Results

device: xiaomi 12s

data: 2025-02-16T14:42:08+08:00

executable: /data/local/tmp/mllm-advanced/bin/MllmArmKernels-GemvOps

num_cpus: 8

mhz_per_cpu: 2016

cpu_scaling_enabled: True

library_version: v1.9.1

library_build_type: release

| Name | Run Name | Run Type | Iterations | Real Time | CPU Time | Time Unit |
| --- | --- | --- | --- | --- | --- | --- |
| hgemv_v1/64 | hgemv_v1/64 | iteration | 2780686 | 251.96931693546975 | 250.27570031280055 | ns |
| hgemv_v1/128 | hgemv_v1/128 | iteration | 259787 | 2724.6463756285575 | 2706.757913213518 | ns |
| hgemv_v1/256 | hgemv_v1/256 | iteration | 74061 | 9397.668529847288 | 9332.190397105089 | ns |
| hgemv_v1/512 | hgemv_v1/512 | iteration | 24115 | 29059.018412183585 | 28843.430437487044 | ns |
| hgemv_v1/1024 | hgemv_v1/1024 | iteration | 7768 | 90997.63581777677 | 90362.65357878472 | ns |
| hgemv_v1/2048 | hgemv_v1/2048 | iteration | 1613 | 436421.36142159 | 433590.805951643 | ns |
| hgemv_v1_4_threads/64 | hgemv_v1_4_threads/64 | iteration | 102352 | 7219.574761619753 | 7189.201803579802 | ns |
| hgemv_v1_4_threads/128 | hgemv_v1_4_threads/128 | iteration | 54500 | 12615.380348389917 | 12568.00952293578 | ns |
| hgemv_v1_4_threads/256 | hgemv_v1_4_threads/256 | iteration | 35307 | 19401.029115431193 | 19327.712125074333 | ns |
| hgemv_v1_4_threads/512 | hgemv_v1_4_threads/512 | iteration | 19320 | 35308.868687298134 | 35185.82329192549 | ns |
| hgemv_v1_4_threads/1024 | hgemv_v1_4_threads/1024 | iteration | 7035 | 88079.66866058529 | 87757.29125799576 | ns |
| hgemv_v1_4_threads/2048 | hgemv_v1_4_threads/2048 | iteration | 2233 | 309394.8025092486 | 308231.98522167455 | ns |
| hgemv_v2_hp/64 | hgemv_v2_hp/64 | iteration | 1583388 | 436.3913147012417 | 433.449012497252 | ns |
| hgemv_v2_hp/128 | hgemv_v2_hp/128 | iteration | 243979 | 2892.2594691112013 | 2872.977223449553 | ns |
| hgemv_v2_hp/256 | hgemv_v2_hp/256 | iteration | 69478 | 10143.085609382468 | 10078.342338581993 | ns |
| hgemv_v2_hp/512 | hgemv_v2_hp/512 | iteration | 19748 | 35547.71105844772 | 35321.20569171563 | ns |
| hgemv_v2_hp/1024 | hgemv_v2_hp/1024 | iteration | 6204 | 116124.1571524473 | 115415.42520954247 | ns |
| hgemv_v2_hp/2048 | hgemv_v2_hp/2048 | iteration | 1328 | 526869.1167153323 | 523468.4442771092 | ns |
| hgemv_v2_hp_4_threads/64 | hgemv_v2_hp_4_threads/64 | iteration | 97246 | 6912.256771581526 | 6886.292906649098 | ns |
| hgemv_v2_hp_4_threads/128 | hgemv_v2_hp_4_threads/128 | iteration | 110149 | 6450.380094283332 | 6422.025084204117 | ns |
| hgemv_v2_hp_4_threads/256 | hgemv_v2_hp_4_threads/256 | iteration | 69819 | 10089.851372114637 | 10046.057892550738 | ns |
| hgemv_v2_hp_4_threads/512 | hgemv_v2_hp_4_threads/512 | iteration | 30377 | 24224.502353428907 | 24119.389340619535 | ns |
| hgemv_v2_hp_4_threads/1024 | hgemv_v2_hp_4_threads/1024 | iteration | 7348 | 94032.91412850484 | 93701.9059608059 | ns |
| hgemv_v2_hp_4_threads/2048 | hgemv_v2_hp_4_threads/2048 | iteration | 1757 | 397772.92601110105 | 396275.82128628326 | ns |
| sgemv_v1/64 | sgemv_v1/64 | iteration | 563133 | 1185.573333492077 | 1177.8322989418157 | ns |
| sgemv_v1/128 | sgemv_v1/128 | iteration | 151755 | 4637.527046901593 | 4603.654937234353 | ns |
| sgemv_v1/256 | sgemv_v1/256 | iteration | 49181 | 14371.735060904564 | 14274.687582603072 | ns |
| sgemv_v1/512 | sgemv_v1/512 | iteration | 14633 | 48877.16333176066 | 48532.13073190733 | ns |
| sgemv_v1/1024 | sgemv_v1/1024 | iteration | 3164 | 222507.39095924483 | 221045.99747155444 | ns |
| sgemv_v1/2048 | sgemv_v1/2048 | iteration | 915 | 762577.64047541 | 757902.0874316933 | ns |
| sgemv_v1_4_threads/64 | sgemv_v1_4_threads/64 | iteration | 90531 | 7436.558338554603 | 7407.769195082342 | ns |
| sgemv_v1_4_threads/128 | sgemv_v1_4_threads/128 | iteration | 60629 | 11626.591993977758 | 11584.022711903526 | ns |
| sgemv_v1_4_threads/256 | sgemv_v1_4_threads/256 | iteration | 32946 | 21282.75086498196 | 21207.710404904945 | ns |
| sgemv_v1_4_threads/512 | sgemv_v1_4_threads/512 | iteration | 15185 | 45205.06802673403 | 45053.788080342456 | ns |
| sgemv_v1_4_threads/1024 | sgemv_v1_4_threads/1024 | iteration | 4496 | 158539.2684542559 | 157956.51979537364 | ns |
| sgemv_v1_4_threads/2048 | sgemv_v1_4_threads/2048 | iteration | 1939 | 360205.59362506145 | 358793.60134089756 | ns |
