cmake_minimum_required (VERSION 3.2)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/generate_data)


add_executable(generate_lorenz generate_lorenz.cpp)
target_link_libraries(generate_lorenz -g
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    ${ARMADILLO_LIBRARIES}
    modelLearning
)


add_executable(generate_rossler generate_rossler.cpp)
target_link_libraries(generate_rossler -g
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    ${ARMADILLO_LIBRARIES}
    modelLearning
)

add_executable(generate_lorenz9d generate_lorenz9d.cpp)
target_link_libraries(generate_lorenz9d -g
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    ${ARMADILLO_LIBRARIES}
    modelLearning
)
