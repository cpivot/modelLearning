cmake_minimum_required (VERSION 3.2)

file(GLOB SRC_FILES
  FT.cpp
  polynome.cpp
  )

if ( ${static_libs} )
  add_library(libFT ${SRC_FILES})
else ()
  add_library(libFT SHARED ${SRC_FILES})
  target_link_libraries(libFT
    -fPIC
    ${ARMADILLO_LIBRARIES}
    )
endif()

set(header ${PROJECT_SOURCE_DIR}/include/FT.hpp
  ${PROJECT_SOURCE_DIR}/include/polynome.hpp
  )

install(TARGETS libFT DESTINATION lib)
install(FILES ${header} DESTINATION include/ft_test)
