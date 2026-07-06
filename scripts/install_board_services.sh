#!/bin/sh
# 板端服务安装脚本：安装 Debian 依赖包、部署 systemd unit、配置文件
# 部署后服务默认不自动开机启动；需要时使用 start_board_services.sh 手动启动
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root" >&2
    exit 1
fi

# 从脚本位置推导部署根目录（脚本在 scripts/ 下，ROOT 为其上一级）
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(dirname "$SCRIPT_DIR")

# 安全校验：/userdata 必须是独立挂载分区
if ! awk '$2 == "/userdata" { found=1 } END { exit found ? 0 : 1 }' /proc/mounts; then
    echo "/userdata is not a mounted filesystem; refusing to install on the root partition" >&2
    exit 1
fi
# 安全校验：/userdata 不能以 noexec 方式挂载（否则程序无法执行）
mount_options=$(awk '$2 == "/userdata" { print $4; exit }' /proc/mounts)
case ",$mount_options," in
    *,noexec,*)
        echo "/userdata is mounted with noexec; executable deployment is not possible" >&2
        exit 1
        ;;
esac
# 安全校验：强制统一部署路径
if [ "$ROOT" != "/userdata/rk3568_inspection_terminal" ]; then
    echo "install root must be /userdata/rk3568_inspection_terminal (got $ROOT)" >&2
    exit 1
fi

# --- 安装 Debian 系统依赖包 ---
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsqlite3-0 sqlite3 libmosquitto1 mosquitto mosquitto-clients

# v4l-utils 仅在系统中不存在 v4l2-ctl 时才安装，避免升级已适配 RKISP 的定制版本
if ! command -v v4l2-ctl >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y v4l-utils
fi

# --- 部署配置文件 ---
install -d -m 0755 /etc/mosquitto/conf.d
install -m 0644 "$ROOT/config/mosquitto.conf" /etc/mosquitto/conf.d/rk3568.conf
install -m 0644 "$ROOT/systemd/rk_video_ai.service" /etc/systemd/system/rk_video_ai.service
install -m 0644 "$ROOT/systemd/rkaiq_3A.service" /etc/systemd/system/rkaiq_3A.service

systemctl daemon-reload

# apt 安装 mosquitto 时可能自动启用服务——统一 stop + disable
# 确保板卡重启后本项目及配套服务不会自动运行
systemctl disable --now rk_video_ai.service 2>/dev/null || true
systemctl disable --now rkaiq_3A.service 2>/dev/null || true
systemctl disable --now mosquitto.service 2>/dev/null || true

# --- 输出部署摘要 ---
echo "Mosquitto: mqtt://127.0.0.1:1883"
echo "External input:   rtsp://192.168.137.1:8554/camera"
echo "WSL publish:      rtsp://192.168.137.1:8554/result"
echo "WSL WebRTC page:  http://192.168.137.1:8889/result"
echo "Autostart: disabled"
echo "Manual start: $ROOT/scripts/start_board_services.sh"
