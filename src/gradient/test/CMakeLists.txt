project(gradient_tests C)

cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_STANDARD 23)

set(project_name gradient_tests)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})

add_executable(${project_name}
               gradient_tests.c
               ../src/gradient.c)

target_include_directories(${project_name} PRIVATE ../src)
target_link_libraries(${project_name} cmocka)
target_link_libraries(${project_name} m)

enable_testing()
add_test(NAME gradient COMMAND ${project_name} 1)
