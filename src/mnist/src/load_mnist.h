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
#include <time.h>

#ifndef STRUCT_MNIST_
#define STRUCT_MNIST_
struct Mnist {
  float *image;
  uint8_t *label;
  uint8_t *one_hot_label;
  _Bool normalize;
};
#endif  // STRUCT_MNIST_

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

int LoadMnist(struct Mnist *train_data, char *dir);

void RandomSampling(struct Mnist *train_data, int data_amount, int image_res,
  int batch_size, float *x_batch, uint8_t *t_batch);

void ReadOnePic(float *x_batch, uint8_t *t_batch,
  struct TwoLayerNeuralNetwork * nn, int site);

#endif  // MNIST_SRC_LOAD_MNIST_H_