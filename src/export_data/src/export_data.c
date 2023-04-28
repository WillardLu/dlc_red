/// -----------------------------------------------------------------
/// @file export_data.c
/// @brief 导出数据
/// @details 导出数据。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-28
/// -----------------------------------------------------------------

#include "export_data.h"

/// @brief 保存学习记录
/// @param data 学习记录数据
void SaveStudyRecord(char * data) {
  // 按照当前日期时间生成文件名称
  char input_file[100] = {0};
  time_t t1 = time(NULL);
  struct tm *tt = localtime(&t1);
  sprintf(input_file, "%d-%d-%d %d:%d:%d.csv\n", tt->tm_year + 1900,
    tt->tm_mon + 1, tt->tm_mday, tt->tm_hour, tt->tm_min, tt->tm_sec);
  // 打开文件并写入记录
  FILE * fp_data = fopen(input_file, "w");
  if (fprintf(fp_data, "%s", data) != -1) {
    printf("学习记录保存成功。\n");
  } else {
    printf("学习记录保存失败。\n");
  }
  fclose(fp_data);
  return;
}