# RK3568 双路视频 AI 巡检终端

RK3568 并行处理一路 Windows 摄像头和一路板载 IMX415：外部 H.264 使用 MPP 硬解码，IMX415 使用 V4L2/DMA-BUF，随后由 RGA 合成、RKNN 三模型推理，并只做一次 MPP H.264 编码发布最终画面。

## 最终拓扑

```text
Windows Integrated Camera
  -> Windows FFmpeg / DirectShow / H.264 720p25
  -> TCP MPEG-TS 单点直连 RK3568:5000
  -> MPP decode -------------------------------+
                                                 +-> 单调时钟同步
RK3568 IMX415 -> V4L2 NV12 DMA-BUF -------------+   -> RGA 1920x1080 mosaic
                                                     -> RKNN COCO/Fire/PPE
                                                     -> MPP H.264 12 Mbps
                                                     -> WSL MediaMTX /result
                                                     -> RTSP / WebRTC
```

输入链路不经过 MediaMTX。WSL MediaMTX 只接收板端 `/result`，因此不会给摄像头输入增加一次发布、拉取和缓冲。

## 已实现能力

- TCP/MPEG-TS 异步监听、MPP H.264 硬解码、断线自动恢复；
- 外部发送端缺席或断流时，IMX415、推理及最终输出继续运行；
- 断流 500 ms 后清除冻结旧帧，离线槽位不再制造伪高延迟；
- IMX415 V4L2 MPLANE、DMA-BUF、RKAIQ 3A；
- 双路 25 FPS 时基与时间戳同步；
- COCO、Fire/Smoke、PPE 三个 INT8 RKNN 模型常驻轮转；
- RGA 等比 cover 合成，1920×1080、25 FPS、12 Mbps；
- SQLite、MQTT、事件录像、温控和运行统计；
- MediaMTX、Mosquitto、RKAIQ 开机自启，主程序按要求手动启动。

## 板端运行环境与依赖

本项目当前适配并验证的板端基线如下。更换系统镜像、内核或 NPU 组件后，需重新确认
RKNPU 驱动、RKNN Runtime、RKNN Server 和 `.rknn` 模型之间的兼容性。

### 硬件与系统

| 项目 | 当前配置 |
|---|---|
| 板卡厂商 | 正点原子（ALIENTEK） |
| 板卡型号 | `ATK-DLRK3568` |
| SoC | Rockchip RK3568，四核 ARM Cortex-A55，内置 1 TOPS NPU |
| 板载摄像头 | IMX415，V4L2 NV12 multiplanar，配合 RKAIQ 3A |
| CPU 架构 | `aarch64` / ARM64 |
| 板端系统 | Debian GNU/Linux 11（bullseye） |
| Linux 内核 | `5.10.160`，Rockchip `rockchip_linux_defconfig` |
| 板级 SDK 配置 | `alientek_rk3568_defconfig`，DTS `rk3568-atk-evb1-ddr4-v10-linux` |
| 默认部署目录 | `/userdata/rk3568_inspection_terminal` |

### NPU 版本基线

| 组件 | 版本 | 说明 |
|---|---:|---|
| RKNPU 内核驱动（NPU Driver） | `0.9.8` | 随当前 `5.10.160` 板端内核/boot 镜像提供 |
| RKNN Runtime（`librknnrt.so`） | `2.3.2` | 项目打包并链接的 Linux/aarch64 运行库 |
| RKNN Server | `2.3.2` | 板端 RKNN 服务版本，应与 Runtime 保持一致 |
| RKNN 模型目标平台 | `rk3568` | 三个 INT8 模型均需以该目标平台导出 |

仓库中的 `librknnrt.so` 来自 RKNPU2 SDK 2.3.2，内部版本为
`2.3.2 (429f97ae6b@2025-04-09T09:09:27)`。不要混用旧版 `librknn_api.so`，
也不要只替换 Runtime 或 Server 中的一个；模型转换工具、Runtime、Server 与驱动不兼容时，
可能出现模型加载失败、版本不匹配或推理异常。

### 主要用户态依赖

| 依赖 | 当前版本/来源 | 用途 |
|---|---|---|
| RKNN Runtime | `2.3.2`，仓库 `lib/rknn/` | NPU 推理 |
| Rockchip MPP | 仓库 `lib/mpp/` | H.264 硬件编解码 |
| Rockchip RGA | im2d API `1.10.1`，仓库 `lib/rga/` | 缩放、色彩转换与画面合成 |
| OpenCV | `4.5.1`，Debian 11/aarch64 | 图像处理与后处理 |
| FFmpeg libraries | `4.3.1`（libavcodec 58.91.100） | MPEG-TS 解复用及媒体处理 |
| SQLite 3 | Debian 11/aarch64 | 本地结构化数据存储 |
| Mosquitto | `2.0.11` | 本机 MQTT Broker 与消息发布 |
| RKAIQ | 随板卡 Rockchip SDK/系统镜像提供 | IMX415 ISP 与 3A |

可在板端复核实际运行环境：

```bash
tr -d '\0' </proc/device-tree/model; echo
cat /etc/os-release
uname -m && uname -r
cat /sys/kernel/debug/rknpu/version 2>/dev/null || \
  cat /proc/rknpu/version 2>/dev/null
/userdata/rk3568_inspection_terminal/scripts/board_hardware_status.sh
```

第三方库的目录结构、获取方式和 RKNN Runtime 校验值见
[`lib/README.md`](lib/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `include/`, `src/` | C++17 板端主程序 |
| `config/sensors.ini` | 输入、同步、推理与编码参数 |
| `windows/` | Windows 原生摄像头直连发布脚本 |
| `wsl/` | MediaMTX 配置与用户级 systemd 服务 |
| `systemd/`, `scripts/` | 板端服务和安装脚本 |
| `docs/` | 完整部署及故障排查手册 |
| `deploy/` | 可部署的 Debian 11/aarch64 `.tar.gz` |

## 默认地址

| 项目 | 默认值 |
|---|---|
| WSL | `192.168.137.1` |
| RK3568 | `192.168.137.2` |
| 外部输入 | RK3568 TCP `5000`，MPEG-TS/H.264 |
| 最终 RTSP | `rtsp://192.168.137.1:8554/result` |
| WebRTC | `http://192.168.137.1:8889/result` |
| 最终编码 | H.264 High、1920×1080、25 FPS、12 Mbps |

## 快速部署

### 1. WSL 只启动 MediaMTX

```bash
cd ~/RK3568_workspace
chmod +x wsl/install_wsl_services.sh
./wsl/install_wsl_services.sh
systemctl --user status rk_mediamtx.service --no-pager
```

### 2. 先启动板端监听

```bash
systemctl start rk_video_ai.service
ss -lntp | grep ':5000 '
```

应看到 `rk3568-main` 监听 `0.0.0.0:5000`。即使 Windows 尚未发送，IMX415 和 `/result` 也会正常输出。

### 3. Windows 原生采集并直连板端

不要通过 USBIP 把集成摄像头挂载给 WSL；该链路已实测出现 UVC `corrupted data` 和重复帧。

管理员 PowerShell：

```powershell
usbipd detach --busid 2-1
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

新开 PowerShell：

```powershell
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
Set-ExecutionPolicy -Scope Process Bypass
.\windows\start_camera_publisher.ps1 `
  -DeviceName "Integrated Camera" -BoardAddress "192.168.137.2"
```

脚本固定输出 H.264/YUV420P/1280×720/25 FPS，1 秒 GOP、无 B 帧，并在断线后每秒重连。

### 4. 编译、打包与板端安装

```bash
cmake --build build-aarch64 -j"$(nproc)"
rm -rf staging-rk3568
DESTDIR="$PWD/staging-rk3568" \
  cmake --install build-aarch64 --prefix /userdata/rk3568_inspection_terminal
tar -C staging-rk3568/userdata -czf \
  deploy/rk3568_inspection_terminal-debian11.tar.gz \
  rk3568_inspection_terminal
sha256sum deploy/rk3568_inspection_terminal-debian11.tar.gz
```

必须传完整压缩包；不要直接复制 `build-aarch64` 中的二进制，它带构建机 RPATH，可能误加载板端旧版 MPP。

```bash
scp deploy/rk3568_inspection_terminal-debian11.tar.gz root@192.168.137.2:/userdata/
ssh root@192.168.137.2
systemctl stop rk_video_ai.service 2>/dev/null || true
tar -xzf /userdata/rk3568_inspection_terminal-debian11.tar.gz -C /userdata
cd /userdata/rk3568_inspection_terminal
./scripts/install_board_services.sh
./scripts/start_board_services.sh
```

## 验收

```bash
# RK3568
ss -lntp | grep ':5000 '
systemctl status rkaiq_3A mosquitto rk_video_ai --no-pager
journalctl -u rk_video_ai -f | grep -E 'RuntimeStats|Metrics|FrameSync|Thermal'

# WSL
ffprobe -v error -rtsp_transport tcp rtsp://127.0.0.1:8554/result
```

浏览器打开 `http://192.168.137.1:8889/result`，应看到一路双画面拼接输出。

### 8.2 录像

```bash
find /userdata/rk3568_inspection_terminal/data/records -maxdepth 1 -type f -ls

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,duration \
  -show_entries packet=pts_time,dts_time,duration_time,flags \
  /userdata/rk3568_inspection_terminal/data/records/<FILE>.mp4
```

首个视频包应包含关键帧标记，PTS/DTS 应单调递增。验证线程解耦：

1. 正常触发一次录像；
2. 停止 WSL/服务器 MediaMTX；
3. 等待 RTSP 线程进入重连；
4. 再次触发事件；
5. 确认本地 MP4 仍正常完成。

### 8.3 SQLite 与 MQTT

```bash
find /userdata/rk3568_inspection_terminal/data/sqlite -name '*.db' -type f
sqlite3 /userdata/rk3568_inspection_terminal/data/sqlite/edge_data_$(date +%Y%m%d).db '.tables'
mosquitto_sub -h 127.0.0.1 -t 'rk3568/#' -v
```

## 9. 常见问题

### RTSP 断开但本地录像也停止

确认 `[mosaic_stream].enable=true` 和 `[record].enabled=true`。`enable_rtsp` 可以关闭，但不能关闭整个 mosaic 管线。

### 没有生成录像

检查推理是否持续产生阳性结果、`confirm_window`/`confirm_min_positive` 是否过严、输出目录是否可写：

```bash
journalctl -u rk_video_ai -b | grep -E 'EventRecorder|Inference|MppEncoder'
df -h /userdata/rk3568_inspection_terminal/data/records
```

### 双路同步丢帧较多

先观察 `[FrameSync] avg_ms/max_ms/drops`，再逐步把 `threshold_ms` 从 `15` 调到 `20` 或 `25`。不要直接设为一个帧周期以上，否则拼接画面的时间一致性会明显下降。

### 实时调度设置失败

查看 `[thread_rt]` 日志和当前权限。systemd 服务以 root 运行时通常可设置 FIFO/RR；手工使用普通用户运行时可能缺少 `CAP_SYS_NICE` 或 `RLIMIT_RTPRIO`。

### 部署后动态库缺失

```bash
export LD_LIBRARY_PATH=/userdata/rk3568_inspection_terminal/lib:$LD_LIBRARY_PATH
ldd /userdata/rk3568_inspection_terminal/bin/rk3568_inspection_terminal | grep 'not found'
```

完整板端与 WSL 操作见 [部署手册](docs/RK3568与WSL部署操作手册.md)。
