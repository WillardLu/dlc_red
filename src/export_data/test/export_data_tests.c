/// -----------------------------------------------------------------
/// @file export_data.c
/// @brief 测试 export_data 模块的功能。
/// @details 测试 export_data 模块的功能。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-28
/// -----------------------------------------------------------------

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmocka.h>

#include "../src/export_data.h"

static void ExportDataTest(void **state) {
  return;
}

int main(int argc, char *argv[]) {
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(ExportDataTest),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
