/// -----------------------------------------------------------------
/// @file mnist_tests.c
/// @brief 测试 mnist 模块的功能。
/// @details 测试 mnist 模块的功能。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-10
/// -----------------------------------------------------------------

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmocka.h>

#include "../src/load_mnist.h"

static void LoadMnistTest(void **state) {
  struct Mnist test_data = {NULL, NULL, NULL, false};
  char image_file[] = "train-images.idx3-ubyte";
  char label_file[] = "train-labels.idx1-ubyte";

  test_data.image = (float *)malloc(sizeof(float) * 60000 * 784);
  test_data.label = (uint8_t *)malloc(sizeof(uint8_t) * 60000);
  test_data.one_hot_label = (uint8_t *)calloc(60000 * 10, sizeof(uint8_t));
  if (test_data.image == NULL || test_data.label == NULL ||
    test_data.one_hot_label == NULL) {
    free(test_data.image);
    free(test_data.label);
    free(test_data.one_hot_label);
    printf("测试数据申请内存失败。\n");
    return;
  }
  // normalize为假时
  if (LoadMnist(&test_data, image_file, label_file) == 0) {
    assert_int_equal(test_data.image[152], 3);
    assert_int_equal(test_data.label[0], 5);
    assert_int_equal(test_data.one_hot_label[5], 1);
  } else {
    printf("LoadMnist函数执行失败。\n");
  }
  test_data.normalize = true;
  // normalize为真时
  if (LoadMnist(&test_data, image_file, label_file) == 0) {
    assert_int_equal(test_data.image[152], 0.011764);
    assert_int_equal(test_data.label[28737], 9);
    assert_int_equal(test_data.one_hot_label[28737*10+9], 1);
  } else {
    printf("LoadMnist函数执行失败。\n");
  }
  free(test_data.image);
  free(test_data.label);
  free(test_data.one_hot_label);
  return;
}

int main(int argc, char *argv[]) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(LoadMnistTest),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}