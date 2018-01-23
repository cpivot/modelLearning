cmake_minimum_required (VERSION 3.2)

file(GLOB SRC_FILES
  *.cpp
  )

if ( ${static_libs} )
  add_library(model ${SRC_FILES})
else ()
  add_library(model SHARED ${SRC_FILES})
  target_link_libraries(model
    -fPIC
    ${ARMADILLO_LIBRARIES}
    base
    optimizer
    repres
    )
endif()
