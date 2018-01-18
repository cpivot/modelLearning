cmake_minimum_required (VERSION 3.2)

file(GLOB SRC_FILES
  *.cpp
  )

if ( ${static_libs} )
  add_library(repres ${SRC_FILES})
else ()
  add_library(repres SHARED ${SRC_FILES})
  target_link_libraries(repres
    -fPIC
    ${ARMADILLO_LIBRARIES}
    base
    )
endif()
