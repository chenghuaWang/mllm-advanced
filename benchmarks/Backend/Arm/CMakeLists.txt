add_executable(MllmArmKernels-ElewiseOps ElewiseOps.cpp)
target_link_libraries(MllmArmKernels-ElewiseOps PRIVATE benchmark::benchmark
                                                        MllmRT MllmArmBackend)

add_executable(MllmArmKernels-GemvOps GemvOps.cpp)
target_link_libraries(MllmArmKernels-GemvOps PRIVATE benchmark::benchmark
                                                     MllmRT MllmArmBackend)

add_executable(MllmArmKernels-ComplexOps ComplexOps.cpp)
target_link_libraries(MllmArmKernels-ComplexOps PRIVATE benchmark::benchmark
                                                        MllmRT MllmArmBackend)

add_executable(MllmArmKernels-GemmOps GemmOps.cpp)
target_link_libraries(MllmArmKernels-GemmOps PRIVATE benchmark::benchmark
                                                        MllmRT MllmArmBackend)
                                                        