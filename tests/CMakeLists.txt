add_executable(main_auto main_auto.cpp)

target_link_libraries(main_auto PRIVATE OpenMP::OpenMP_CXX
                                PRIVATE AutoStrategizer
                                PRIVATE numa)

add_subdirectory( old_tests )