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
/// @param image_file 图像文件名称
/// @param label_file 标签文件名称
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
  uint8_t *image = NULL; // 用于中转图像数据的数组  
  image = (uint8_t *)malloc(sizeof(uint8_t) * 60000 * 784);
  if ( image == NULL ) {
    printf("用于中转图像数据的数组内存分配失败！\n");
    fclose(fp_image);
    return -1;
  }
  fseek(fp_image, 16, SEEK_SET); // 跳过文件头部的非数据区
  fread(image, 1, 60000 * 784, fp_image);
  fclose(fp_image);
  // 根据normalize值转换图像数据
  float normalize = train_data->normalize == true ? 255.0f : 1.0f;
  for (int i = 0; i < 60000 * 784; i++) {
    train_data->image[i] = image[i] / normalize;
  }
  free(image);
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