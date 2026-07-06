#!/bin/sh
# MediaMTX 流媒体服务器启动脚本（运行于 WSL）
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# 可通过 MEDIAMTX_BIN 环境变量指定可执行文件路径，默认使用 PATH 中的 mediamtx
MEDIAMTX_BIN=${MEDIAMTX_BIN:-mediamtx}

# exec 替换当前 shell 进程为 MediaMTX，节省进程槽位，且信号直接传递
exec "$MEDIAMTX_BIN" "$SCRIPT_DIR/mediamtx.yml"
