/// -----------------------------------------------------------------
/// @file load_mnist.c
/// @brief 载入 MNIST 训练数据
/// @details 载入 MNIST 数据集的训练用图片与标签数据。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-10
/// -----------------------------------------------------------------

#include "load_mnist.h"

/// @brief 载入训练数据
/// @param train_data 要载入的训练数据结构
/// @param dir 文件所在目录（相对于执行位置）
/// @return 成功: 0；失败：-1
int LoadMnist(struct Mnist *train_data, char *dir) {
  FILE *fp_image = NULL;
  FILE *fp_label = NULL;
  char image_file[100] = {0};
  char label_file[100] = {0};

  // 如果没有输入命令行参数，会造成内存出错，所以加上此判断。
  if (dir != NULL) {
    strcat(image_file, dir);
    strcat(label_file, dir);
  }
  strcat(image_file, "train-images.idx3-ubyte");
  strcat(label_file, "train-labels.idx1-ubyte");
  // 读取image_file文件的内容
  fp_image = fopen(image_file, "rb");
  if (fp_image == NULL) {
    printf("图像训练数据文件打开失败。\n");
    return -1;
  }
  
  static uint8_t image[60000 * 784]; // 用于中转图像数据的数组

  fseek(fp_image, 16, SEEK_SET); // 跳过文件头部的非数据区
  fread(image, 1, 60000 * 784, fp_image);
  fclose(fp_image);
  // 根据normalize值转换图像数据
  float normalize = train_data->normalize == true ? 255.0f : 1.0f;
  for (int i = 0; i < 60000 * 784; i++) {
    train_data->image[i] = image[i] / normalize;
  }
  // 读取label_file文件的内容
  fp_label = fopen(label_file, "rb");
  if (fp_label == NULL) {
    printf("标签训练数据文件打开失败。\n");
    return -1;
  }
  fseek(fp_label, 8, SEEK_SET); // 跳过文件头部的非数据区
  fread(train_data->label, sizeof(uint8_t), 60000, fp_label);
  fclose(fp_label);
  // 将标签转换成one hot形式，并保存到数组中
  for (int i = 0; i < 60000; i++) {
    train_data->one_hot_label[i * 10 + train_data->label[i]] = 1;
  }
  return 0;
}

/// @brief 随机取样
/// @param train_data 要随机取样的数据集
/// @param data_amount 数据集中的数据数量
/// @param image_res 图片精度
/// @param batch_size 每次取的图片数量
/// @param x_batch 存放取样的图片数据
/// @param y_batch 存放取样的标签数据
void RandomSampling(struct Mnist *train_data, int data_amount, int image_res,
  int batch_size, float *x_batch, uint8_t *t_batch) {
  for (int i = 0; i < batch_size; i++) {
    srand(rand() * (unsigned int)time(NULL));
    int sample_index = rand() % data_amount;
    for (int k = 0; k < image_res; k++) {
      x_batch[i * image_res + k] =
        train_data->image[sample_index * image_res + k];
    }
    for (int k = 0; k < 10; k++) {
      t_batch[i * 10 + k] =
        train_data->one_hot_label[sample_index * 10 + k];
    }
  }
  return;
}

/// @brief 从数据集中读取一张图片的数据
/// @param x_batch 图片数据集（本项目中指的是随机抽样的图片数据集）
/// @param t_batch 标签数据集（本项目中指的是随机抽样的标签数据集）
/// @param nn 两层神经网络结构体
/// @param site 在数据集中的位置
void ReadOnePic(float *x_batch, uint8_t *t_batch,
  struct TwoLayerNeuralNetwork * nn, int site) {
  for (int i = 0; i < 784; i++) {
    nn->x[i] = x_batch[site * 784 + i];
  }
  for (int i = 0; i < 10; i++) {
    nn->t[i] = t_batch[site * 10 + i];
  }
  return;
}