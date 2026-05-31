#!/bin/bash
# check_checkpoint.sh
# 用法: ./check_checkpoint.sh [文件名关键字]
# 默认搜索整个 openpi cache 目录

SEARCH_KEYWORD=${1:-pi05_base}  # 如果没传参数，默认搜索 pi05_base
CACHE_DIR="/bak"

echo "Searching for files containing '$SEARCH_KEYWORD' in $CACHE_DIR ..."
echo "--------------------------------------"

# 使用 find 搜索
find "$CACHE_DIR" -type f -name "*$SEARCH_KEYWORD*" -exec ls -lh {} \;

echo "--------------------------------------"
echo "Search complete."