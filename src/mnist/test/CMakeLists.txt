project(mnist_tests C)

cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_STANDARD 23)

set(project_name mnist_tests)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})

add_executable(${project_name}
               mnist_tests.c
               ../src/load_mnist.c)

target_include_directories(${project_name} PRIVATE ../src)
target_link_libraries(${project_name} cmocka)

enable_testing()
add_test(NAME LoadMnist COMMAND ${project_name} 1)