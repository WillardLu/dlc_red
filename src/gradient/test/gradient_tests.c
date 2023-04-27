/// -----------------------------------------------------------------
/// @file gradient.c
/// @brief 测试 gradient 模块的功能。
/// @details 测试 gradient 模块的功能。
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

#include "../src/gradient.h"

static void GradientTest(void **state) {
  return;
}

int main(int argc, char *argv[]) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(GradientTest),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}