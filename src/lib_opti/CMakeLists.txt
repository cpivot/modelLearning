cmake_minimum_required (VERSION 3.2)

file(GLOB SRC_FILES
  *.cpp
  )

if ( ${static_libs} )
  add_library(optimizer ${SRC_FILES})
else ()
  add_library(optimizer SHARED ${SRC_FILES})
  target_link_libraries(optimizer
    -fPIC
    ${ARMADILLO_LIBRARIES}
    base
    )
endif()
