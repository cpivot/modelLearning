cmake_minimum_required (VERSION 3.2)

file(GLOB SRC_FILES
  *.cpp
  )

if ( ${static_libs} )
  add_library(base ${SRC_FILES})
else ()
  add_library(base SHARED ${SRC_FILES})
  target_link_libraries(base
    -fPIC
    ${ARMADILLO_LIBRARIES}
    )
endif()
