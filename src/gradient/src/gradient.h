/// -----------------------------------------------------------------
/// @file gradient.h
/// @brief gradient 模块的头文件。
/// @details 计算梯度。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-27
/// -----------------------------------------------------------------
#ifndef GRADIENT_SRC_NUMERICAL_DIFFERENTIATION_H_
#define GRADIENT_SRC_NUMERICAL_DIFFERENTIATION_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void Predict(struct TwoLayerNeuralNetwork *nn);

float CrossEntropyError(struct TwoLayerNeuralNetwork * nn);

void NumericalDifferentiation(struct TwoLayerNeuralNetwork *nn);

#endif  // GRADIENT_SRC_NUMERICAL_DIFFERENTIATION_H_