cmake_minimum_required (VERSION 3.2)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


add_executable(test_prog test_arma.cpp)
target_link_libraries(test_prog -g
    ${ARMADILLO_LIBRARIES}
    modelLearning
      )

add_executable(test_poly test_polynome.cpp)
target_link_libraries(test_poly -g
  ${ARMADILLO_LIBRARIES}
  modelLearning
  )
