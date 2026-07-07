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

完整板端与 WSL 操作见 [部署手册](docs/RK3568与WSL部署操作手册.md)，事件录像设计与实现边界见 [事件预录制评估](docs/事件预录制方案评估与实现说明.md)。
