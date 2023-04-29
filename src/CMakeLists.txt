set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(dlc_red
               main.c
               mnist/src/load_mnist.c
               gradient/src/gradient.c
               initialization_parameters/src/initialization_parameters.c
               export_data/src/export_data.c)
target_link_libraries(dlc_red m)
