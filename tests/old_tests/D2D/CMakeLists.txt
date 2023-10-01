add_executable(d2d_test_11 d2d_test_11.cpp)

target_link_libraries(d2d_test_11 PRIVATE OpenMP::OpenMP_CXX
                                  PRIVATE numa)

add_executable(d2d_test_4 d2d_test_4.cpp)

target_link_libraries(d2d_test_4 PRIVATE OpenMP::OpenMP_CXX
                                 PRIVATE numa)