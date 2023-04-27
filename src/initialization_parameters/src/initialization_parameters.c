/// -----------------------------------------------------------------
/// @file initialization_parameters.c
/// @brief 初始化参数
/// @details 初始化权重参数。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-10
/// -----------------------------------------------------------------

#include "initialization_parameters.h"

// @brief 生成权重参数
int CreateWeight(float * w, int size, float min, float max,
  float weight_init_std) {
  srand(rand() * (unsigned int)time(NULL));
  for (int i = 0; i < size; i++) {
    w[i] = weight_init_std * (((float)rand() / RAND_MAX) * (max * 2) + min);
  }
  return 0;
}

// @brief 初始化权重参数
void InitWeight(struct TwoLayerNeuralNetwork * nn, float min, float max,
  float weight_init_std) {
  CreateWeight(nn->w1, nn->input_size * nn->hidden_size, min, max,
    weight_init_std);
  CreateWeight(nn->w2, nn->hidden_size * nn->output_size, min, max,
    weight_init_std);
  return;
}