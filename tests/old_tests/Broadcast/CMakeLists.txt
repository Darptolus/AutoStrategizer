add_executable(broadcast_22 broadcast_22.cpp)

target_link_libraries(broadcast_22 PRIVATE OpenMP::OpenMP_CXX
                                   PRIVATE numa)
 