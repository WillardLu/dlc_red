set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR})
add_executable(mnist_tests
               mnist_tests.c
               ../src/load_mnist.c)
target_include_directories(mnist_tests PRIVATE ../src)
target_link_libraries(mnist_tests cmocka)

enable_testing()
add_test(NAME LoadMnist COMMAND mnist_tests 1)