add_executable(main_auto main_auto.cpp)

target_link_libraries(main_auto PRIVATE OpenMP::OpenMP_CXX
                                PRIVATE AutoStrategyzer
                                PRIVATE numa)