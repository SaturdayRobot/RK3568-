# RK3568 双路视频异构三模型推理与事件预录制

面向 RK3568、Linux 5.10、Debian 11 的边缘视频程序。系统接收一路外部 RTSP 和一路 IMX415，按区域给不同视频源路由不同模型：RTSP 执行人员检测，V4L2 执行猫、狗检测。两条采集与预处理管线并行推进，单 NPU 上的推理调用统一串行仲裁；最终画面只经过一次 MPP H.264 编码，同时分发给 RTSP 推流和本地事件录像。

主要能力：

- 外部 RTSP：FFmpeg 解封装、MPP 硬解码、RGA/RKNN 处理；
- IMX415：V4L2 MPLANE、MMAP、DMA-BUF、RGA 转换；
- COCO、Fire/Smoke、PPE 三类 RKNN 模型常驻，按视频源轮转调度；
- 双路按采集单调时间配对，输出同步误差和丢帧统计；
- 拼接画面单次 MPP H.264 编码；
- RTSP 与录像使用独立线程、独立有界队列；
- AI 事件支持前录、后录、防抖、连续事件合并；
- fragmented MP4、SQLite、MQTT、磁盘保留策略；
- systemd、自恢复、硬件 watchdog、温控降级和运行统计。

## 1. 运行架构

### 1.1 代码框架图

```text
src/main.cpp
  |
  `-- AppInitializer -------------------------------- 应用装配与生命周期
       |
       |-- InferenceService ------------------------- 三模型路由与单 NPU 仲裁
       |     |-- RTSP -> COCO / PPE轮转
       |     |-- V4L2 -> COCO / Fire轮转
       |     `-- rknn_lite / YOLOv8 DFL postprocess
       |
       |-- VisibleRtspPipeline ---------------------- 外部 RTSP 管线
       |     |-- StreamLoader (FFmpeg + MPP decode)
       |     |-- RgaPreprocessor
       |     |-- process thread (RGA + frame callback)
       |     `-- latest-only inference thread
       |
       |-- V4l2CameraPipeline ----------------------- IMX415 管线
       |     |-- capture thread (V4L2 DQBUF/QBUF)
       |     |-- process thread (RGA + rotate)
       |     `-- inference thread
       |
       |-- FrameHub --------------------------------- 帧共享与双路同步队列
       |
       |-- MosaicStreamPipeline --------------------- 时间配对、缩放与拼接
       |     `-- EncodedMediaService ---------------- 单次 MPP 编码与包分发
       |           |-- RTSP packet queue -> RTSP output thread
       |           `-- record packet queue -> EventRecorder
       |                                      |-- encoded ring buffer
       |                                      |-- event state machine
       |                                      `-- fragmented MP4 / retention
       |
       |-- InferenceResultDispatcher ---------------- 推理结果异步落库
       |-- SqliteStore ------------------------------- SQLite/WAL/批量写入
       |-- MqttUploader ------------------------------ MQTT 上传与重连
       |-- BoardStatusCollector ---------------------- 板端状态采集
       `-- NpuThermalManager ------------------------- 温度监控与推理降级
```

关键目录：

| 路径 | 内容 |
|---|---|
| `include/`、`src/` | C++17 头文件与实现 |
| `config/sensors.ini` | 视频、录像、同步、存储、MQTT、线程配置 |
| `model/rk3568/` | RK3568 RKNN 模型 |
| `cmake/toolchains/` | Debian 11 aarch64 交叉编译工具链 |
| `systemd/` | 主服务与 RKAIQ 3A 服务 |
| `scripts/` | 板端安装及硬件状态脚本 |
| `wsl/` | MediaMTX 配置和启动脚本 |
| `docs/` | 部署手册及事件录像设计说明 |

### 1.2 视频与事件数据流向图

```text
外部 RTSP
  -> FFmpeg demux -> MPP decode -> DMA-BUF -> RGA显示帧 -----------+
                                      `-> latest-only异步RKNN ------+
                                                               |
IMX415 -> V4L2 DQBUF -> DMA-BUF -> RGA显示/旋转 -------------------+
                                      `-> latest-only异步RKNN ------+
                                                               v
                FrameHub 每路同步小队列
                capture_mono_ns 最近时间配对
                过旧帧丢弃 + 同步误差统计
                               |
                               v
                    RGA 等比缩放与双路拼接
                    最终坐标系抗锯齿OSD
                               |
                     latest-only 画面队列
                               |
                               v
                 BGR -> RGA NV12 -> MPP H.264
                          单次硬件编码
                               |
                    shared_ptr<EncodedPacket>
                     +---------+----------+
                     |                    |
                     v                    v
             RTSP 有界包队列       录像有界包队列
                     |                    |
               RTSP 输出线程       编码包环形缓存
               断线独立重连              |
                                          +-- AI/人工/传感器事件
                                          |      -> 5 选 3 防抖
                                          |      -> 延长截止时间
                                          v
                                 最近 IDR + 前录 + 后录
                                          |
                                fragmented MP4 + SQLite
```

RTSP 网络阻塞不会持有录像锁；磁盘写入不会运行在采集、拼接或编码线程。录像队列溢出时会结束异常片段并等待新 IDR，RTSP 队列丢包只会让 RTSP 消费者重新同步。

### 1.3 时间体系

| 时间 | 用途 |
|---|---|
| `CLOCK_MONOTONIC` / `steady_clock` | 双路同步、前录、后录、事件合并、超时 |
| V4L2 单调时间戳 | IMX415 真实采集时刻；驱动无效时在 `DQBUF` 后补时间 |
| `CLOCK_REALTIME` / `system_clock` | 文件名、日志、SQLite、界面显示 |
| 90 kHz PTS/DTS | RTSP 和 MP4 播放时间轴 |

外部 RTSP 无法获得统一的传感器曝光时刻，目前使用解码帧交付时刻。需要严格相机级同步时，应在摄像头侧增加 PTP、硬件触发或源 PTS 到统一时钟的映射。

## 2. 三模型按源路由

默认模型：

```text
model/yolov8n_coco_int8.rknn
model/yolov8n_fire_smoke_int8.rknn
model/yolov8n_sh17_ppe_int8.rknn
```

RK3568 只有一个 NPU 核心。RTSP 与 V4L2 可以并行采集、解码和预处理，但所有 `rknn_run` 通过同一互斥仲裁。两路都使用 latest-only 推理任务，视频发布线程不等待 NPU。默认路由为：

```ini
[visible_inference]
enable = true
model_coco = ../model/yolov8n_coco_int8.rknn
model_fire = ../model/yolov8n_fire_smoke_int8.rknn
model_ppe = ../model/yolov8n_sh17_ppe_int8.rknn
external_models = coco,ppe
imx415_models = coco,fire
external_models_per_frame = 1
imx415_models_per_frame = 1
```

每个请求只执行一个模型：A路在COCO/PPE间轮转，B路在COCO/Fire间轮转，并短期复用其他模型的最近结果。这样将默认模型运行需求减半，并避免任一路长时间占用NPU。

### 2.1 模型与量化格式

当前部署的是三个YOLOv8n INT8 RKNN模型，输入均为`1×3×640×640`，采用Rockchip优化的三尺度9输出DFL格式。COCO模型输出80类，Fire/Smoke输出2类，SH17-PPE输出17类；运行时再按视频源过滤业务类别。正式部署重新量化时，应使用现场代表性校准图像。

预处理使用保持比例的YOLOv8 letterbox（填充值114），后处理使用class-aware DIoU-NMS；各模型的置信度与NMS IoU阈值可在`[visible_inference]`独立调整。显示层还会对相邻结果做IoU匹配和平滑，但不会让过期检测框长期停留。

## 3. 配置

主配置文件为 `config/sensors.ini`。systemd 默认通过相对路径 `../config/sensors.ini` 加载；也可以设置环境变量：

```bash
export EDGE_SENSOR_CONFIG=/path/to/sensors.ini
```

### 3.1 视频输入与输出

```ini
[video_source_external]
enable = true
url = rtsp://192.168.137.1:8554/camera
loader_rtsp_transport = tcp
inference_interval_ms = 150

[video_source_imx415]
enable = true
device = /dev/video0
width = 1920
height = 1080
fps = 30
pixel_format = NV12
rotation = 270
inference_interval_ms = 150

[mosaic_stream]
enable = true
enable_rtsp = true
input_mode = side_by_side
width = 1280
height = 720
fps = 25
bitrate = 10000000
stream_queue_size = 1
rtsp_packet_queue_size = 30
preserve_aspect_ratio = true
aspect_mode = cover
draw_overlay = true

[stream_output]
url = rtsp://192.168.137.1:8554/result
```

`enable_rtsp=false` 只关闭网络输出，MPP 编码与事件录像仍继续工作。

### 3.2 事件预录制

```ini
[record]
enabled = true
pre_seconds = 3
post_seconds = 5
cache_seconds = 5
packet_queue_size = 512
confirm_window = 5
confirm_min_positive = 3
output_dir = ../data/records
max_files = 200
max_storage_mb = 4096
min_free_space_mb = 512
```

- `cache_seconds` 建议不小于 `pre_seconds + 2`，用于覆盖 GOP 和安全余量；
- person、cat、dog 任一目标数量大于 0 即为一次阳性检测；
- 默认最近 5 次推理至少 3 次阳性才触发；
- 持续事件只延长当前片段，不重复创建文件；
- 正常文件为 `YYYYMMDD_HHMMSS_Event.mp4`，录制中为 `.mp4.part`；
- 文件清理同时受数量、总容量和最低剩余空间约束。

录像目录按服务工作目录解析，默认实际位置为：

```text
/userdata/rk3568_inspection_terminal/data/records
```

### 3.3 双路同步

```ini
[sync]
enabled = true
threshold_ms = 15
queue_depth = 4
```

30 fps 建议从 `10~20 ms` 开始调试。阈值过小会增加丢帧，过大会增加两路画面时间差。每 300 对同步帧输出一次：

```text
[FrameSync] pairs=300 avg_ms=4.2 max_ms=12.7 drops=3,1
```

### 3.4 SQLite 与 MQTT

```ini
[storage]
enable = true
base_dir = ../data/sqlite
retention_days = 30
wal = true
sync = NORMAL
write_batch_size = 50
write_flush_ms = 500

[mqtt]
enable = true
host = 127.0.0.1
port = 1883
topic = rk3568/rk3568_node_001/data
qos = 1
```

录像完成信息以 `event_record` 类型进入现有 SQLite 上传队列。生产环境启用跨机器 TLS 时，应配置可信 CA 并保持 `tls_insecure=false`。

### 3.5 温控策略

`thermal-monitor` 每秒读取 `/sys/class/thermal/thermal_zone1/temp`。65/75/85°C 分别进入 light、medium、heavy，推理节流系数依次为 0.70/0.40/0.10；heavy 状态允许直接跳过推理，但采集、显示、编码和录像继续运行。降温超过 4°C 回差后才退级，避免阈值附近反复切换。预测温控在 55°C 以上且升温速率达到 2°C/s 时提前降级。所有阈值和策略均可在 `[thermal]` 修改。

## 4. 编译与打包

### 4.1 交叉编译

在项目根目录执行：

```bash
cmake -S . -B build-rk3568 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/rk3568-debian11.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF

cmake --build build-rk3568 -j$(nproc)
file build-rk3568/rk3568_inspection_terminal
```

产物应显示为 `ARM aarch64`。完整干净构建：

```bash
cmake --build build-rk3568 --clean-first -j$(nproc)
```

### 4.2 生成部署目录和压缩包

```bash
rm -rf staging-rk3568
DESTDIR="$PWD/staging-rk3568" \
  cmake --install build-rk3568 --prefix /userdata/rk3568_inspection_terminal

tar -C staging-rk3568/userdata -czf \
  deploy/rk3568_inspection_terminal-debian11.tar.gz \
  rk3568_inspection_terminal

sha256sum deploy/rk3568_inspection_terminal-debian11.tar.gz \
  > deploy/SHA256SUMS
```

部署目录包含程序、配置、模型、运行库、systemd 文件、脚本和文档。

## 5. 部署

### 5.1 WSL/服务器启动 MediaMTX

编辑 `wsl/mediamtx.yml`，使 `webrtcAdditionalHosts` 使用板端和浏览器可访问的地址，然后启动：

```bash
MEDIAMTX_BIN=/path/to/mediamtx ./wsl/start_mediamtx.sh
```

默认端点：

| 功能 | 地址 |
|---|---|
| 外部视频输入 | `rtsp://192.168.137.1:8554/camera` |
| RK3568 拼接输出 | `rtsp://192.168.137.1:8554/result` |
| WebRTC 页面 | `http://192.168.137.1:8889/result` |
| WHEP | `http://192.168.137.1:8889/result/whep` |

可用测试流模拟外部摄像头：

```bash
ffmpeg -re -f lavfi -i "testsrc2=size=1280x720:rate=30" \
  -an -c:v libx264 -preset veryfast -tune zerolatency \
  -pix_fmt yuv420p -g 30 -bf 0 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/camera
```

### 5.2 安装到 RK3568

```bash
scp deploy/rk3568_inspection_terminal-debian11.tar.gz root@<BOARD_IP>:/tmp/
ssh root@<BOARD_IP>

rm -rf /tmp/rk3568_inspection_terminal
tar -xzf /tmp/rk3568_inspection_terminal-debian11.tar.gz -C /tmp \
  rk3568_inspection_terminal/scripts/clean_board_install.sh
/tmp/rk3568_inspection_terminal/scripts/clean_board_install.sh
rm -rf /tmp/rk3568_inspection_terminal
tar -xzf /tmp/rk3568_inspection_terminal-debian11.tar.gz -C /userdata
cd /userdata/rk3568_inspection_terminal
./scripts/install_board_services.sh
```

安装脚本会配置 Mosquitto、主程序服务和可用时的 RKAIQ 3A 服务。启动前检查：

```bash
uname -m
v4l2-ctl -d /dev/video0 --get-fmt-video
ls -lh /userdata/rk3568_inspection_terminal/model/*.rknn
ldd /userdata/rk3568_inspection_terminal/bin/rk3568_inspection_terminal | grep 'not found'
```

最后一条正常时没有输出。

## 6. 启动与停止

推荐启动顺序：MediaMTX、外部 `/camera` 视频、板端 Mosquitto、RKAIQ、主程序。

### 6.1 systemd 启动

```bash
/userdata/rk3568_inspection_terminal/scripts/start_board_services.sh
systemctl status rk_video_ai --no-pager
journalctl -u rk_video_ai -f
```

常用命令：

```bash
systemctl restart rk_video_ai
systemctl stop rk_video_ai
systemctl start rk_video_ai
journalctl -u rk_video_ai -b --no-pager
```

修改 `/userdata/rk3568_inspection_terminal/config/sensors.ini` 后需要重启服务。若要覆盖配置路径，新建：

```bash
cat >/etc/default/rk3568-inspection <<'EOF'
EDGE_SENSOR_CONFIG=/userdata/rk3568_inspection_terminal/config/sensors.ini
EOF
systemctl restart rk_video_ai
```

### 6.2 前台启动

前台调试必须从 `bin` 目录启动，确保默认相对路径正确：

```bash
systemctl stop rk_video_ai
cd /userdata/rk3568_inspection_terminal/bin
EDGE_SENSOR_CONFIG=../config/sensors.ini ./rk3568_inspection_terminal
```

按 `Ctrl+C` 触发优雅退出。

## 7. 线程模型与统计

### 7.1 主要线程

| 线程名 | 角色 | 队列/阻塞隔离 |
|---|---|---|
| `rtsp-loader` | 外部流拉取、解封装和解码 | 解码 ready queue |
| `rtsp-process` | 外部帧转换和发布 | 不等待NPU、拼接或编码 |
| `rtsp-infer` | 外部流轮转模型推理 | latest-only硬件帧；不阻塞显示 |
| `imx415-cap` | V4L2 DQBUF | latest buffer，旧帧可替换 |
| `imx415-proc` | RGA 转换/旋转和发布 | 不等待 NPU |
| `imx415-infer` | IMX415轮转模型推理 | latest-only推理帧；不阻塞采集/显示 |
| `mosaic-loop` | 双路配对、等比拼接、最终OSD | latest-only 编码输入 |
| `media-encoder` | BGR->NV12、MPP H.264 | 唯一编码生产者 |
| `rtsp-output` | RTSP mux、发送和重连 | 独立编码包队列 |
| `event-record` | 环形缓存、状态机、MP4、清理 | 独立编码包队列 |
| `infer-dispatch` | 推理结果采样和 SQLite 分发 | 1024 深度有界队列 |
| `mqtt-upload` | MQTT 发布和重连 | 与 SQLite 写线程分离 |
| `thermal-monitor` | 温度采样和降级控制 | 独立周期线程 |

RK3568 的 4 个 CPU 都是 Cortex-A55，不存在 big.LITTLE 大小核。当前按任务类型分工，并为 CPU0 保留系统、中断和其它服务的运行空间：

| CPU | 应用线程分工 |
|---|---|
| CPU0 | 主程序不绑定；留给内核、IRQ、systemd 等系统负载 |
| CPU1 | RTSP 拉取/输出、事件录像、MQTT、温控 |
| CPU2 | V4L2 采集、NPU 推理提交、推理结果分发 |
| CPU3 | 图像处理、双路拼接、NV12 转换和 MPP 编码提交 |

线程运行参数使用 `[thread_rt_<role>]` 覆盖全局 `[thread_rt]`。全局使用普通调度，只有采集与视频关键路径显式使用 `SCHED_RR`：

```ini
[thread_rt]
process_cpus = 1-3
policy = other
priority = 0
cpus = 1-3

[thread_rt_imx415_capture]
policy = rr
priority = 62
cpus = 2

[thread_rt_imx415_process]
policy = rr
priority = 60
cpus = 3

[thread_rt_media_encoder]
policy = rr
priority = 54
cpus = 3

[thread_rt_rtsp_output]
policy = rr
priority = 40
cpus = 1

[thread_rt_event_recorder]
policy = other
priority = 0
nice = 5
cpus = 1
```

实时调度需要 root 或 `CAP_SYS_NICE`。systemd 服务配置了 `LimitRTPRIO=99` 和 `LimitMEMLOCK=infinity`；`event-record`、MQTT 和温控使用普通调度并降低 nice，避免后台 I/O 抢占视频关键路径。每个线程启动后都会输出最终生效值：

```text
[thread_rt] applied role=media_encoder thread=media-encoder cpus=3 policy=RR priority=54 nice=0
```

### 7.2 查看逐线程调度与 CPU

```bash
PID=$(pidof rk3568_inspection_terminal)

ps -L -p "$PID" \
  -o pid,tid,psr,cls,rtprio,pri,ni,stat,pcpu,comm

top -H -p "$PID"
taskset -pc "$PID"

journalctl -u rk_video_ai -b \
  | grep '\[thread_rt\] applied'
```

查看指定线程的上下文切换与调度信息：

```bash
for TID in /proc/$PID/task/*; do
  echo "=== ${TID##*/} $(cat "$TID/comm") ==="
  grep -E 'voluntary_ctxt_switches|nonvoluntary_ctxt_switches|Cpus_allowed_list' "$TID/status"
done
```

### 7.3 程序运行统计

主线程每 5 秒输出一次 `[RuntimeStats]` JSON，并同时输出管线、温控和板端状态：

```bash
journalctl -u rk_video_ai -f \
  | grep --line-buffered -E 'RuntimeStats|Metrics|FrameSync|Thermal|EventRecorder|RtspOutput'
```

主要字段：

| 字段 | 含义 |
|---|---|
| `thermal.temp/level/throttle` | NPU 温度、降级等级和节流系数 |
| `board` | CPU、内存、磁盘、网络、摄像头、模型耗时 |
| `storage.pending_upload` | SQLite 待上传记录数 |
| `storage.last_flush_us/flush_count` | 最近刷盘耗时和累计次数 |
| `mqtt.publish_success/publish_failed` | MQTT 发布成功与失败次数 |
| `mqtt.reconnect_attempts` | MQTT 重连次数 |
| `inference_dispatch.dropped` | 推理分发队列累计丢弃数 |
| `inference_dispatch.sample_ms` | 当前动态采样窗口 |
| `pipelines.external_rtsp.fps` | 外部输入处理帧率 |
| `frames_in/processed/dropped` | 输入、处理和丢弃帧数 |
| `decode_queue_depth/limit` | 外部解码队列当前深度与上限 |
| `decode_drop_total` | 解码队列累计丢帧 |
| `last_infer_us/last_total_us` | 最近推理和整帧处理耗时 |
| `inference_fps` | 实际完成的推理请求数/秒，与是否检测到目标无关 |
| `mosaic.output_fps/encoded_fps/sent_fps` | 拼接提交、硬件编码和MediaMTX发送速率 |
| `mosaic.frame_age_ms` | 最新源帧到进入编码器的板端帧龄 |
| 画面全局状态栏 | 输出FPS、采集到编码帧龄、丢帧率及CPU/NPU/RGA/DDR占用 |
| `board.models[]` | coco、fire、ppe 各上下文调用次数及最近/平均/最大耗时 |
| `person_count/cat_count/dog_count` | SQLite 推理摘要中的三类目标数量 |

同步统计由 `[FrameSync]` 输出；录像或 RTSP 包队列溢出会输出明确告警。系统硬件统计脚本：

```bash
/userdata/rk3568_inspection_terminal/scripts/board_hardware_status.sh
watch -n 1 /userdata/rk3568_inspection_terminal/scripts/board_hardware_status.sh
```

## 8. 功能验证

### 8.1 视频和推流

```bash
ffprobe -rtsp_transport tcp rtsp://192.168.137.1:8554/camera
ffprobe -rtsp_transport tcp rtsp://192.168.137.1:8554/result
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
