cmake_minimum_required (VERSION 3.2)

if (BUILD_SUB_LIBS)
  add_subdirectory(lib_repres)
  add_subdirectory(lib_opti)
  add_subdirectory(lib_base)
endif()


set(src lib_repres/FT.cpp
        lib_repres/poly_element.cpp
        lib_repres/kernel_element.cpp
        lib_base/polynome.cpp
        lib_opti/adam.cpp
  )

set(header  ${PROJECT_SOURCE_DIR}/include/lib_repres/lib_repres.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/poly_element.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/kernel_element.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/FT.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_base/lib_base.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_base/polynome.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_opti/lib_opti.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_opi/adam.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_dyn/def_system.hpp
    )

#create a libray containing everything!

file(GLOB SRC_FILES ${src})

if ( ${static_libs} )
  add_library(modelLearning ${SRC_FILES})
else ()
  add_library(modelLearning SHARED ${SRC_FILES})
  target_link_libraries(modelLearning
    -fPIC
    ${ARMADILLO_LIBRARIES}
    )
endif()


install(TARGETS modelLearning DESTINATION lib)
install(FILES ${header} DESTINATION include/modelLearning)
