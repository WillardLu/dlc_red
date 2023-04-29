/// -------------------------------------------------------------------------
/// @file main.c
/// @brief 主程序
/// @details 训练识别手写数字的神经网络。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-16
/// -------------------------------------------------------------------------

#include "mnist/src/load_mnist.h"
#include "initialization_parameters/src/initialization_parameters.h"
#include "gradient/src/gradient.h"
#include "export_data/src/export_data.h"

// 这里使用宏定义是因为如果不使用的话，在后续的定义中就只能使用动态分配。
#define DATA_MOUNT 60000 // 训练数据集大小
#define BATCH_SIZE 100   // 每次迭代的图像数量
#define IMAGE_RES 784    // 图像的分辨率
#define INPUT_SIZE 784   // 输入层神经元数量
#define HIDDEN_SIZE 50   // 隐藏层神经元数量
#define OUTPUT_SIZE 10   // 输出层神经元数量

// 运行程序时，后面跟上训练数据文件所在目录，不需要输入文件名
int main(int argc, char *argv[]) {
  // 设置时间效率检测参数
  clock_t start, end, time1, time2;
  start = clock();
  // 载入MNIST训练数据集
  static float train_image[DATA_MOUNT * IMAGE_RES];
  uint8_t label_m[DATA_MOUNT] = {0};
  uint8_t one_hot_label_m[DATA_MOUNT * 10] = {0};
  static float test_image[10000 * IMAGE_RES];
  uint8_t label_t[10000] = {0};
  uint8_t one_hot_label_t[10000 * 10] = {0};
  struct Mnist train_data = {train_image, label_m, one_hot_label_m, test_image,
    label_t, one_hot_label_t, true};
  printf("------------------ 载入训练数据 ------------------\n\n");
  if (LoadMnist(&train_data, argv[1]) == -1) {
    //free(train_data.image);
    return -1;
  }
  // 初始化两层神经网络结构变量
  // 这里不使用动态分配内存，是因为动态分配是在堆上分配，速度没有在栈上快。
  // 还好这部分所需内存少，栈空间完全能满足。
  printf("------------------ 初始化参数 --------------------\n\n");
  float x_n[784] = {0};
  float w1_n[784 * 50] = {0};
  float b1_n[50] = {0};
  float a1_n[50] = {0};
  float z1_n[50] = {0};
  float w2_n[50 * 10] = {0};
  float b2_n[10] = {0};
  float y_n[10] = {0};
  float w1_g[INPUT_SIZE * HIDDEN_SIZE] = {0};
  float b1_g[HIDDEN_SIZE] = {0};
  float w2_g[HIDDEN_SIZE * OUTPUT_SIZE] = {0};
  float b2_g[OUTPUT_SIZE] = {0};
  const float h = 1e-4; // 微小变化值
  uint8_t t[10] = {0}; // one hot 形式单张图片数据的标签
  struct TwoLayerNeuralNetwork nn = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    x_n, w1_n, b1_n, a1_n, z1_n, w2_n, b2_n, y_n, w1_g, b1_g, w2_g, b2_g, t,
    h};
  const float weight_init_std = 0.01; // 权重标准差
  // 注意，这里设置的最小值与最大值是指参数乘以weight_init_std之前的值。
  InitWeight(&nn, -1, 1, weight_init_std);

  // 设置超参数
  const int iters_num = 10000; // 迭代次数
  const float learning_rate = 0.1; // 学习率
  // 平均每个 epoch 的重复次数
  const int iter_per_epoch = DATA_MOUNT / BATCH_SIZE;
  
  // 随机抽取的训练数据
  float x_batch[BATCH_SIZE * IMAGE_RES] = {0};
  uint8_t t_batch[BATCH_SIZE * 10] = {0};
  // 误差总和
  float sum_y = 0;
  // 用于保存到文件的误差数据的字符串
  char study_records[40000] = "x,y\xA";
  char per_record[100] = {0};
  // 用于保存精度数据的字符串
  char accuracy_records[40000] = "x,y1,y2\xA 0,0,0\xA";
  char per_test_record[100] = {0};
  float train_acc = 0;
  float test_acc = 0;

  // ------------------ 开始学习 ------------------
  printf("------------------ 开始学习 ----------------------\n\n");
  for (int i = 0; i < iters_num; ++i) {
    time1 = clock();
    // 1、随机抽取训练数据
    RandomSampling(&train_data, DATA_MOUNT, IMAGE_RES, BATCH_SIZE, x_batch,
      t_batch);
    // 2、每个批次
    for (int j = 0; j < BATCH_SIZE; j++) {
      // 1）从抽取的图片数据中读入一个图片的数据
      ReadOnePic(x_batch, t_batch, &nn, j);
      // 2）计算单个图片的梯度
      NumericalDifferentiation(&nn);
      // 3）更新参数
      UpdateParam(&nn, learning_rate);
      // 4）计算损失函数
      Predict(&nn);
      sum_y += CrossEntropyError(&nn);
    }
    sum_y = sum_y / BATCH_SIZE;
    time2 = clock();
    printf("%d、损失函数：%f，花费时间：%f\n", i, sum_y,
      (double)(time2-time1)/CLOCKS_PER_SEC);
    sprintf(per_record, "%d,%.4f\xA", i, sum_y);
    strcat(study_records, per_record);
    // 计算每个 epoch 的识别精度
    if ((i + 1) % iter_per_epoch == 0) {
      train_acc = Accuracy(&nn, train_data.image, train_data.one_hot_label, 60000);
      test_acc = Accuracy(&nn, train_data.image_t, train_data.one_hot_label_t, 10000);
      sprintf(per_test_record, "%d,%.4f,%.4f\xA", (int)(i / iter_per_epoch) + 1, train_acc, test_acc);
      strcat(accuracy_records, per_test_record);
    }
  }
  //free(train_data.image);
  end = clock();
  printf("------------------ 学习结束 ----------------------\n\n");
  printf("共花费时间：%f\n", (float)(end-start) / CLOCKS_PER_SEC);
  // ------------------ 结束学习 ------------------

  // 将数据写入文件中
  SaveRecord(study_records, "study");
  SaveRecord(accuracy_records, "accuracy");
  return 0;
}