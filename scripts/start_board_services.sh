#!/bin/sh
# 板端服务启动脚本：启动 MQTT Broker / 3A 服务 / 主推理服务（仅本次会话，不设开机自启）
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root" >&2
    exit 1
fi

# 启动 Mosquitto MQTT Broker
systemctl start mosquitto.service

# 3A 服务需要 rkaiq_3A_server 可执行文件和 IMX415 IQ 校准文件同时存在才启动
if [ -x /usr/bin/rkaiq_3A_server ] && \
   [ -f /etc/iqfiles/imx415_CMK-OT1522-FG3_CS-P1150-IRC-8M-FAU.json ]; then
    systemctl start rkaiq_3A.service
fi

# 启动主推理服务
systemctl start rk_video_ai.service

# 显示服务运行状态（完整输出，不分页）
systemctl --no-pager --full status rk_video_ai.service
