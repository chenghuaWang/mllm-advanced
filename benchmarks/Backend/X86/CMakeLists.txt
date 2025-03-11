add_executable(MllmX86Kernels-GemvOps GemvOps.cpp)
target_link_libraries(MllmX86Kernels-GemvOps PRIVATE benchmark::benchmark
                                                     MllmRT MllmX86Backend)