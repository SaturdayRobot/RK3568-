#!/bin/sh
# 板端安装清理脚本：停止服务、删除 systemd unit、清理部署文件
# 仅清理本项目文件，不卸载 Debian 系统包，避免影响板上其他程序
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root" >&2
    exit 1
fi

APP_ROOT=/userdata/rk3568_inspection_terminal

# --- 停止并禁用所有服务 ---
# 2>/dev/null || true 确保服务不存在或已停止时不会因 set -e 而退出
systemctl disable --now rk_video_ai.service 2>/dev/null || true
systemctl disable --now rkaiq_3A.service 2>/dev/null || true
systemctl disable --now mosquitto.service 2>/dev/null || true

# --- 删除 systemd unit 和配置文件 ---
rm -f /etc/systemd/system/rk_video_ai.service
rm -f /etc/systemd/system/rkaiq_3A.service
rm -f /etc/mosquitto/conf.d/rk3568.conf
rm -f /etc/default/rk3568-inspection
systemctl daemon-reload
systemctl reset-failed rk_video_ai.service rkaiq_3A.service 2>/dev/null || true

# --- 删除 /userdata 独立部署目录（含录像和 SQLite 数据）---
rm -rf "$APP_ROOT"

# --- 清理 /opt 旧版部署目录 ---
rm -rf /opt/rk3568_inspection_terminal

# --- 清理历史平铺版 /opt 下的已知文件 ---
rm -f /opt/bin/rk3568_inspection_terminal
rm -f /opt/config/sensors.ini /opt/config/mosquitto.conf
rm -f /opt/model/yolov8n_coco_int8.rknn
rm -f /opt/model/yolov8n_fire_smoke_int8.rknn
rm -f /opt/model/yolov8n_sh17_ppe_int8.rknn
rm -f /opt/model/labels/coco_trimmed.txt
rm -f /opt/model/labels/sh17_ppe_trimmed.txt
rm -f /opt/model/labels/fire_smoke_labels.txt
rm -f /opt/scripts/install_board_services.sh
rm -f /opt/scripts/board_hardware_status.sh
rm -f /opt/docs/RK3568与WSL部署操作手册.md
rm -f /opt/docs/事件预录制方案评估与实现说明.md
rm -f /opt/systemd/rk_video_ai.service /opt/systemd/rkaiq_3A.service
rm -rf /opt/data/records /opt/data/sqlite

# FFmpeg 旧版 .so 文件（含符号链接和带版本号的文件）
rm -f /opt/lib/libavcodec.so /opt/lib/libavcodec.so.58 /opt/lib/libavcodec.so.58.91.100
rm -f /opt/lib/libavformat.so /opt/lib/libavformat.so.58 /opt/lib/libavformat.so.58.45.100
rm -f /opt/lib/libavutil.so /opt/lib/libavutil.so.56 /opt/lib/libavutil.so.56.51.100
rm -f /opt/lib/libswresample.so /opt/lib/libswresample.so.3 /opt/lib/libswresample.so.3.7.100
rm -f /opt/lib/libswscale.so /opt/lib/libswscale.so.5 /opt/lib/libswscale.so.5.7.100
rm -f /opt/lib/libopencv_core.so /opt/lib/libopencv_core.so.4.5 /opt/lib/libopencv_core.so.4.5.1
rm -f /opt/lib/libopencv_imgproc.so /opt/lib/libopencv_imgproc.so.4.5 /opt/lib/libopencv_imgproc.so.4.5.1
rm -f /opt/lib/librga.so /opt/lib/librknnrt.so /opt/lib/librockchip_mpp.so.1

# rmdir: 仅删除空目录，其他程序的文件不会被误删
rmdir /opt/bin /opt/config /opt/model/labels /opt/model /opt/scripts /opt/docs \
      /opt/systemd /opt/data /opt/lib 2>/dev/null || true

echo "RK3568 inspection terminal files removed; related services are disabled."
