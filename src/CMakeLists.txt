set( AutoStrategizer_src
     autoStrategizer.cpp
     )     

set( AutoStrategizer_inc
     ${CMAKE_SOURCE_DIR}/include/autoStrategizer.hpp
     )

add_library( AutoStrategizer SHARED ${AutoStrategizer_src} ${AutoStrategizer_inc} )

set_target_properties( AutoStrategizer PROPERTIES LINKER_LANGUAGE CXX
                                        PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/autoStrategizer.hpp")