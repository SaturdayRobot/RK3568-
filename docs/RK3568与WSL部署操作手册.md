# RK3568、Windows 与 WSL 部署操作手册

本文对应最终直连版本。默认 WSL 为 `192.168.137.1`，RK3568 为 `192.168.137.2`，板端为 Debian 11/aarch64。

## 1. 架构与职责

```text
Windows Camera -> Windows FFmpeg -> TCP/MPEG-TS -> RK3568:5000
                                                    |
IMX415 -> RKAIQ/V4L2/DMA-BUF -----------------------+
                                                    v
                         时间同步 -> RGA 合成 -> RKNN 推理
                         -> MPP H.264 1080p25/12Mbps
                         -> WSL MediaMTX /result
                         -> RTSP 或 WebRTC 客户端
```

- Windows 原生 DirectShow 负责读取摄像头和一次 H.264 编码；
- RK3568 是 TCP 服务端，监听 `5000`，直接接收 MPEG-TS；
- MediaMTX 不在输入路径，只承接板端最终 `/result`；
- 外部输入断开不阻塞 IMX415 和最终输出，发送端恢复后自动重连。

这样可避开 WSL USBIP/UVC 已实测的损坏帧、重复帧和成批交付问题。

## 2. 地址与端口

| 功能 | 地址/端口 |
|---|---|
| Windows 到板端输入 | TCP `192.168.137.2:5000` |
| 板端合成输出 | `rtsp://192.168.137.1:8554/result` |
| WebRTC 页面 | `http://192.168.137.1:8889/result` |
| WebRTC 媒体 | UDP `8189` |
| 板端 MQTT | `127.0.0.1:1883` |

Windows 防火墙需允许 FFmpeg 访问 TCP 5000；WSL/Windows 需允许 TCP 8554、8889 与 UDP 8189。

## 3. WSL 部署 MediaMTX

```bash
command -v mediamtx
chmod +x ~/.local/bin/mediamtx
cd ~/RK3568_workspace
chmod +x wsl/install_wsl_services.sh
./wsl/install_wsl_services.sh
systemctl --user status rk_mediamtx.service --no-pager
ss -lntup | grep -E '8554|8889|8189'
```

旧版 WSL 摄像头发布服务不再使用，安装脚本会禁用它。也可手动清理：

```bash
systemctl --user disable --now rk_camera_publisher.path 2>/dev/null || true
systemctl --user stop rk_camera_publisher.service 2>/dev/null || true
```

若希望 Windows 登录后仍自动启动 WSL 用户服务：

```bash
sudo loginctl enable-linger "$USER"
systemctl --user enable rk_mediamtx.service
```

## 4. Windows 摄像头直连

### 4.1 解除 USBIP 挂载

管理员 PowerShell：

```powershell
usbipd list
usbipd detach --busid 2-1
```

`2-1` 需按实际 BUSID 修改。摄像头改由 Windows 原生驱动使用，不应再出现在 WSL `/dev/video0`。

### 4.2 安装 FFmpeg 并确认设备名

```powershell
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

关闭并重开 PowerShell：

```powershell
ffmpeg -version
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

记下视频设备名，默认脚本使用 `Integrated Camera`。

### 4.3 启动板端和发送脚本

先在 RK3568 执行：

```bash
systemctl start rk_video_ai.service
ss -lntp | grep ':5000 '
```

再在项目目录的 Windows PowerShell 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows\start_camera_publisher.ps1 `
  -DeviceName "Integrated Camera" `
  -BoardAddress "192.168.137.2" `
  -Port 5000
```

若脚本先启动，连接会被拒绝，但会每秒重试，不需要手动重启。

### 4.4 Windows 任务计划开机启动（可选）

创建“用户登录时”任务，程序填写 `powershell.exe`，参数填写：

```text
-NoProfile -ExecutionPolicy Bypass -File "C:\path\RK3568_workspace\windows\start_camera_publisher.ps1" -DeviceName "Integrated Camera" -BoardAddress "192.168.137.2"
```

是否自启发送器由 Windows 任务计划控制；板端主程序仍按要求保持手动启动。

## 5. 编译与打包

```bash
cd ~/RK3568_workspace
cmake --build build-aarch64 -j"$(nproc)"
rm -rf staging-rk3568
DESTDIR="$PWD/staging-rk3568" \
  cmake --install build-aarch64 --prefix /userdata/rk3568_inspection_terminal
tar -C staging-rk3568/userdata -czf \
  deploy/rk3568_inspection_terminal-debian11.tar.gz \
  rk3568_inspection_terminal
sha256sum deploy/rk3568_inspection_terminal-debian11.tar.gz
readelf -d staging-rk3568/userdata/rk3568_inspection_terminal/bin/rk3568_inspection_terminal \
  | grep -E 'RPATH|RUNPATH'
```

RPATH 必须为 `$ORIGIN/../lib`。构建目录二进制不能直接部署，否则可能出现 `undefined symbol: mpp_buffer_sync_begin_f`。

## 6. 板端安装

```bash
scp deploy/rk3568_inspection_terminal-debian11.tar.gz root@192.168.137.2:/userdata/
ssh root@192.168.137.2
findmnt /userdata
systemctl stop rk_video_ai.service 2>/dev/null || true
tar -xzf /userdata/rk3568_inspection_terminal-debian11.tar.gz -C /userdata
cd /userdata/rk3568_inspection_terminal
./scripts/install_board_services.sh
./scripts/start_board_services.sh
```

安装脚本会启用 `mosquitto.service` 与 `rkaiq_3A.service`。`rk_video_ai.service` 被安装但保持 disabled，由用户显式启动。

关键配置：

```ini
[video_source_external]
type = tcp_mpegts
url = tcp://0.0.0.0:5000?listen=1&tcp_nodelay=1
loader_max_ready_depth = 8
inference_interval_ms = 150

[video_source_imx415]
device = /dev/video0
fps = 25
pixel_format = NV12
color_space = bt601_full
rotation = 270

[stream_output]
url = rtsp://192.168.137.1:8554/result

[mosaic_stream]
width = 1920
height = 1080
fps = 25
bitrate = 12000000
aspect_mode = cover

[sync]
enabled = true
threshold_ms = 45
queue_depth = 4
source_stale_ms = 500
```

## 7. 自启策略

```bash
systemctl is-enabled mosquitto.service
systemctl is-enabled rkaiq_3A.service
systemctl is-enabled rk_video_ai.service
```

预期分别为 `enabled`、`enabled`、`disabled`。WSL 的 `rk_mediamtx.service` 为 enabled；Windows FFmpeg 是否自启由任务计划决定。

## 8. 验收

板端：

```bash
ss -lntp | grep ':5000 '
systemctl status rkaiq_3A mosquitto rk_video_ai --no-pager
ldd /userdata/rk3568_inspection_terminal/bin/rk3568_inspection_terminal | grep 'not found'
journalctl -u rk_video_ai -f \
  | grep --line-buffered -E 'Metrics:rtsp|RuntimeStats|FrameSync|Thermal'
```

Windows 发送后应满足：

- `external_rtsp_online=true`、`imx415_online=true`；
- 两路均约 25 FPS；
- `decode_queue_depth` 接近 0，`decode_drop_total=0`；
- PPE、COCO、Fire 模型计数持续增长；
- `mosaic.output_fps/encoded_fps/sent_fps` 约 25；
- `frame_drops=0`、`rtsp_packet_drops=0`；
- `frame_age_ms` 不随运行时间持续增长。

WSL：

```bash
ffprobe -v error -rtsp_transport tcp -read_intervals '%+5' -count_frames \
  -select_streams v:0 \
  -show_entries stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames \
  -of default=nw=1 rtsp://127.0.0.1:8554/result
```

预期 H.264 High、YUV420P、1920×1080、25/1 FPS。浏览器可打开 `http://192.168.137.1:8889/result`。

## 9. 画面与延迟说明

- Windows 输入使用 YUV420P、25 FPS、1 秒 GOP、无 B 帧、8 Mbps；
- IMX415 使用 RKAIQ 3A 和 `bt601_full`，用于修复偏色及过曝；
- 两路固定槽位、等比 cover，不进行非等比拉伸；
- 断流超过 500 ms 后显示离线占位，不长期复用冻结帧；
- 最终输出只编码一次，1080p25 使用 12 Mbps；
- 顶部 `DROP` 是采集、解码、编码和网络发送的真实增量丢弃，不把同步候选淘汰算作网络丢包。

## 10. 故障排查

### Windows 显示 `Connection refused`

```bash
systemctl status rk_video_ai.service --no-pager
ss -lntp | grep ':5000 '
journalctl -u rk_video_ai -n 100 --no-pager
```

启动板端后 Windows 脚本会在 1 秒内重试。

### 外部画面离线，但 IMX415 正常

```powershell
Test-NetConnection 192.168.137.2 -Port 5000
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

检查 Windows 摄像头设备名、FFmpeg 日志、防火墙和网络。

### `rk_video_ai.service not found`

服务名是 `rk_video_ai.service`，不是 `rkvideo_ai.service`。重新运行 `scripts/install_board_services.sh`。

### `undefined symbol: mpp_buffer_sync_begin_f`

误部署了构建目录二进制。重新部署完整 `.tar.gz`，确认 RPATH 为 `$ORIGIN/../lib`。

### IMX415 绿色或过曝

```bash
systemctl status rkaiq_3A.service --no-pager
ls -l /etc/iqfiles/imx415_CMK-OT1522-FG3_CS-P1150-IRC-8M-FAU.json
journalctl -u rkaiq_3A -n 100 --no-pager
```

### 延迟或丢帧持续增加

查看 `decode_queue_depth`、`decode_drop_total`、`frame_drops`、`rtsp_packet_drops` 和 `frame_age_ms`。Windows FFmpeg 应直连 5000，不应再发布 `/camera`；板端必须使用最新配置和 500 ms 旧帧失效。

## 11. 升级

```bash
scp deploy/rk3568_inspection_terminal-debian11.tar.gz root@192.168.137.2:/userdata/
ssh root@192.168.137.2
systemctl stop rk_video_ai.service
tar -xzf /userdata/rk3568_inspection_terminal-debian11.tar.gz -C /userdata
systemctl daemon-reload
systemctl start rk_video_ai.service
```

升级不会主动删除 `data/records` 与 `data/sqlite`。
