set( AutoStrategyzer_src
     autoStrategyzer.cpp
     )     

set( AutoStrategyzer_inc
     ${CMAKE_SOURCE_DIR}/include/autoStrategyzer.hpp
     )

add_library( AutoStrategyzer SHARED ${AutoStrategyzer_src} ${AutoStrategyzer_inc} )

set_target_properties( AutoStrategyzer PROPERTIES LINKER_LANGUAGE CXX
                                        PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/autoStrategyzer.hpp")