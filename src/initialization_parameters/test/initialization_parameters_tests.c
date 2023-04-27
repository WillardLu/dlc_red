/// -----------------------------------------------------------------
/// @file initialization_parameters_tests.c
/// @brief 测试 initialization_parameters 模块的功能。
/// @details 测试 initialization_parameters 模块的功能。
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

#include "../src/initialization_parameters.h"

static void InitParamTest(void **state) {
  // 为测试准备测试数据
  float x_n[784] = {0};
  float w1_n[784 * 50] = {0};
  float b1_n[50] = {0};
  float a1_n[50] = {0};
  float z1_n[50] = {0};
  float w2_n[50 * 10] = {0};
  float b2_n[10] = {0};
  float y_n[10] = {0};
  float w1_g[784 * 50] = {0};
  float b1_g[50] = {0};
  float w2_g[50 * 10] = {0};
  float b2_g[10] = {0};
  const float h = 1e-4; // 微小变化值
  uint8_t t[10] = {0}; // one hot 形式单张图片数据的标签
  struct TwoLayerNeuralNetwork nn = {784, 50, 10, x_n, w1_n, b1_n, a1_n, z1_n,
    w2_n, b2_n, y_n, w1_g, b1_g, w2_g, b2_g, t, h};
  // normalize为真时
  InitWeight(&nn, -1, 1, 0.01);
  assert_in_range((int)(nn.w1[11] * 1000 + 1000), 0, 2000);
  assert_in_range((int)(nn.w1[39100] * 1000 + 1000), 0, 2000);
  assert_in_range((int)(nn.w2[11] * 1000 + 1000), 0, 2000);
  assert_in_range((int)(nn.w2[498] * 1000 + 1000), 0, 2000);
  return;
}

int main(int argc, char *argv[]) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(InitParamTest),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}