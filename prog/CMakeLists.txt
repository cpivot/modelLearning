cmake_minimum_required (VERSION 3.2)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/prog)

add_subdirectory(generate_data)


add_executable(learn_model learn_model.cpp)
target_link_libraries(learn_model -g
    ${ARMADILLO_LIBRARIES}
    modelLearning
      )

add_executable(learn_rosenbrock learn_rosenbrock.cpp)
target_link_libraries(learn_rosenbrock -g
    ${ARMADILLO_LIBRARIES}
    modelLearning
    )
