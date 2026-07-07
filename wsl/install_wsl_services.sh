#!/bin/sh
# 安装并启动WSL侧MediaMTX（用户级systemd，无需sudo）。
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v mediamtx >/dev/null 2>&1; then
    echo "mediamtx not found in PATH" >&2
    exit 1
fi
systemctl --user link "$SCRIPT_DIR/rk_mediamtx.service" 2>/dev/null || true
systemctl --user daemon-reload
# 旧架构的USBIP摄像头发布器必须停用，避免继续占用摄像头和CPU。
systemctl --user disable --now rk_camera_publisher.path 2>/dev/null || true
systemctl --user stop rk_camera_publisher.service 2>/dev/null || true
systemctl --user enable --now rk_mediamtx.service
systemctl --user --no-pager --full status rk_mediamtx.service || true
