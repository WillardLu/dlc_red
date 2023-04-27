/// -----------------------------------------------------------------
/// @file load_mnist.h
/// @brief load_mnist 模块的头文件。
/// @details 读取 MNIST 训练数据。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-10
/// -----------------------------------------------------------------
#ifndef MNIST_SRC_LOAD_MNIST_H_
#define MNIST_SRC_LOAD_MNIST_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef STRUCT_MNIST_
#define STRUCT_MNIST_
struct Mnist {
  float *image;
  uint8_t *label;
  uint8_t *one_hot_label;
  _Bool normalize;
};
#endif  // STRUCT_MNIST_

int LoadMnist(struct Mnist *train_data, char *dir);

#endif  // MNIST_SRC_LOAD_MNIST_H_