cmake_minimum_required(VERSION 3.20)

project(dlc_red VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)

# 使用enable_testing()后，在build目录中会生成一个Testing文件夹，
# 里面包含测试函数的实现。然后在build目录下执行ctest命令才生效。
enable_testing()
add_subdirectory(src)
# add_subdirectory(test)