# 依赖库目录

本目录用于存储项目所需的第三方依赖库。

## 目录结构

```
lib/
├── rknn/            # RKNN相关库
│   ├── include/     # RKNN头文件
│   └── lib/         # RKNN库文件
├── mpp/             # MPP相关库
│   ├── include/     # MPP头文件
│   └── lib/         # MPP库文件
├── rga/             # RGA相关库
│   ├── include/     # RGA头文件
│   └── lib/         # RGA库文件
└── README.md        # 目录说明文档
```

## 依赖库获取

### 1. RKNN库

- 当前版本：RKNPU2 Runtime 2.3.2（Linux/aarch64）
- 官方来源：`airockchip/rknn-toolkit2` 的 `v2.3.2` 标签
- 运行时只链接 `librknnrt.so`，不要再混用旧版 `librknn_api.so`
- `rknn_api.h` 必须与 `librknnrt.so` 来自同一 SDK 版本

### 2. MPP库

- **从Rockchip官方获取**：
  - 访问Rockchip开发者网站
  - 下载Media Process Platform库
  - 解压到`lib/mpp/`目录

### 3. RGA库

- **从Rockchip官方获取**：
  - 访问Rockchip开发者网站
  - 下载Rockchip Graphics Accelerator库
  - 解压到`lib/rga/`目录

## 库文件结构

### RKNN库文件

```
lib/rknn/
├── include/
│   └── rknn_api.h
└── lib/
    └── librknnrt.so
```

### MPP库文件

```
lib/mpp/
├── include/
│   ├── mpp_buffer.h
│   ├── mpp_decoder.h
│   └── ...
└── lib/
    └── librockchip_mpp.so
```

### RGA库文件

```
lib/rga/
├── include/
│   └── rga.h
└── lib/
    └── librga.so
```

## 注意事项

- 确保 `.rknn` 模型以 `target_platform="rk3568"` 导出
- 确保板端 RKNPU 驱动、RKNN Runtime 和模型编译器版本兼容
- 库文件应具有可执行权限
- 当前 `librknnrt.so` SHA256：`d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`
