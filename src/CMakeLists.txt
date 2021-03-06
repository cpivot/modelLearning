cmake_minimum_required (VERSION 3.2)

if (BUILD_SUB_LIBS)
  add_subdirectory(lib_repres)
  add_subdirectory(lib_opti)
  add_subdirectory(lib_base)
  add_subdirectory(lib_model)
endif()


set(src lib_repres/FT.cpp
        lib_repres/poly_element.cpp
        lib_repres/kernel_element.cpp
        lib_base/polynome.cpp
        lib_opti/*.cpp
        lib_model/model.cpp
  )

set(header  ${PROJECT_SOURCE_DIR}/include/lib_repres/lib_repres.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/poly_element.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/kernel_element.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_repres/FT.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_base/lib_base.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_base/polynome.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_opti/lib_opti.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_opti/adam.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_opti/adadelta.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_model/lib_model.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_model/model.hpp
            ${PROJECT_SOURCE_DIR}/include/lib_model/loss_function.hpp
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
