/// -----------------------------------------------------------------
/// @file initialization_parameters.h
/// @brief initialization_parameters 模块的头文件。
/// @details 参数初始化。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-27
/// -----------------------------------------------------------------
#ifndef MNIST_SRC_LOAD_MNIST_H_
#define MNIST_SRC_LOAD_MNIST_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef STRUCT_TWOLAYERNEURALNETWORK_
#define STRUCT_TWOLAYERNEURALNETWORK_
// 两层神经网络结构
struct TwoLayerNeuralNetwork {
  int input_size;
  int hidden_size;
  int output_size;
  float * x;
  float * w1;
  float * b1;
  float * a1;
  float * z1;
  float * w2;
  float * b2;
  float * y;
  float *w1_g;
  float *b1_g;
  float *w2_g;
  float *b2_g;
  uint8_t *t;
  float h;
};
#endif  // STRUCT_TWOLAYERNEURALNETWORK_

void InitWeight(struct TwoLayerNeuralNetwork * nn, float min, float max,
  float weight_init_std);

#endif  // MNIST_SRC_LOAD_MNIST_H_