/// -----------------------------------------------------------------
/// @file export_data.h
/// @brief export_data 模块的头文件。
/// @details 导出数据。
/// @copyright Copyright (c) 2023, 陆巍，All rights reserved.
/// @author 陆巍
/// @date 2023-04-28
/// -----------------------------------------------------------------
#ifndef EXPORT_DATA_SRC_EXPORT_DATA_H_
#define EXPORT_DATA_SRC_EXPORT_DATA_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void SaveRecord(char *data, char *file_name);

#endif  // EXPORT_DATA_SRC_EXPORT_DATA_H_
