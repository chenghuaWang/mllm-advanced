add_executable(MllmArmKernels-ElewiseOps ElewiseOps.cpp)
target_link_libraries(MllmArmKernels-ElewiseOps PRIVATE benchmark::benchmark
                                                        MllmRT MllmArmBackend)
