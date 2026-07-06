#!/usr/bin/env python3
"""将项目的 YOLOv5 训练权重导出为 RK3568 专用的 RKNN 模型。

C++ 后处理器期望三个经过 sigmoid 激活的 NCHW 特征图作为输出，
因此本导出器故意跳过 YOLOv5 内置的 decode/NMS 路径，直接输出原始特征图。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types
from dataclasses import dataclass

import onnx
import torch
from rknn.api import RKNN

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    """描述一个待导出模型的所有配置：名称、权重文件、校准图像目录。"""
    name: str
    checkpoint: pathlib.Path
    image_dir: pathlib.Path


# 需要导出的四个 YOLOv5 模型定义
MODEL_SPECS = (
    ModelSpec("silo_volume", ROOT / "model/Silo_Volume/best.pt", ROOT / "model/Silo_Volume/images/train"),
    ModelSpec("waste_sack_level", ROOT / "model/Waste_Sack_Level/Waste_Sack_Level.pt",
              ROOT / "model/Waste_Sack_Level/images/train"),
    ModelSpec("handle", ROOT / "model/handle/best.pt", ROOT / "model/handle/images/train"),
    ModelSpec("traffic_led", ROOT / "model/traffic_led/traffic_led.pt",
              ROOT / "model/traffic_led/images/train"),
)


def raw_detect_forward(self, features):
    """自定义检测头前向：对每层特征图执行 1x1 卷积 + sigmoid，不进行 decode/NMS。
    输出格式为 NCHW，供 C++ postprocess.cpp 使用。"""
    return tuple(torch.sigmoid(self.m[i](features[i])) for i in range(self.nl))


def load_model(checkpoint: pathlib.Path, yolov5_root: pathlib.Path):
    """加载 YOLOv5 模型，并将检测头替换为原始特征图输出模式。"""
    sys.path.insert(0, str(yolov5_root))
    # Windows 训练的 checkpoint 元数据含 WindowsPath，在 Linux 下反序列化时需映射为 PosixPath
    pathlib.WindowsPath = pathlib.PosixPath
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # 优先使用 EMA 权重（泛化性能更好），转 float32 并设为 eval 模式
    model = (checkpoint_data.get("ema") or checkpoint_data["model"]).float().eval()
    detect = model.model[-1]
    detect.inplace = False  # 禁用 inplace，避免 ONNX 导出时出现不可导出的原地操作
    detect.forward = types.MethodType(raw_detect_forward, detect)
    return model


def export_onnx(spec: ModelSpec, yolov5_root: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """步骤一：将 PyTorch 模型导出为 ONNX 格式，并对输出形状做双重校验。"""
    model = load_model(spec.checkpoint, yolov5_root)
    output_path = output_dir / f"{spec.name}.onnx"
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    with torch.inference_mode():
        outputs = model(dummy)

    # 校验 PyTorch 输出形状：3 个检测层，每层通道数 = 3*(5+类别数)
    expected_channels = 3 * (5 + len(model.names))
    expected_shapes = (
        (1, expected_channels, 80, 80),
        (1, expected_channels, 40, 40),
        (1, expected_channels, 20, 20),
    )
    actual_shapes = tuple(tuple(t.shape) for t in outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"{spec.name}: unexpected PyTorch outputs {actual_shapes}, expected {expected_shapes}")

    # opset_version=12：RKNN Toolkit 对该版本兼容性最佳
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["images"],
        output_names=["output0", "output1", "output2"],
        opset_version=12,
        do_constant_folding=True,
    )

    # 二次校验：ONNX 计算图输出形状
    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    onnx_shapes = tuple(
        tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in graph.graph.output
    )
    if onnx_shapes != expected_shapes:
        raise RuntimeError(f"{spec.name}: unexpected ONNX outputs {onnx_shapes}, expected {expected_shapes}")

    print(f"[ONNX] {spec.name}: classes={model.names}, outputs={onnx_shapes}, file={output_path}")
    return output_path


def make_dataset(spec: ModelSpec, output_dir: pathlib.Path, limit: int) -> pathlib.Path:
    """创建 RKNN 量化校准数据集文件。

    通过符号链接避免路径中包含空格（RKNN 数据集解析器按空格分隔多输入模型路径）。
    图像数量超过 limit 时使用线性插值均匀采样，确保覆盖整个数据集分布。
    """
    images = sorted(
        path for path in spec.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise RuntimeError(f"{spec.name}: no calibration images found in {spec.image_dir}")
    if len(images) > limit:
        indices = [round(i * (len(images) - 1) / (limit - 1)) for i in range(limit)]
        images = [images[i] for i in indices]

    calibration_dir = output_dir / "calibration" / spec.name
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_images = []
    for index, source in enumerate(images):
        link = calibration_dir / f"{index:03d}{source.suffix.lower()}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source.resolve())
        calibration_images.append(link)

    dataset_path = output_dir / f"{spec.name}_dataset.txt"
    dataset_path.write_text(
        "\n".join(str(path.absolute()) for path in calibration_images) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def export_rknn(spec: ModelSpec, onnx_path: pathlib.Path, output_dir: pathlib.Path,
                dataset_limit: int) -> pathlib.Path:
    """步骤二：将 ONNX 模型转换为 RK3568 专用 INT8 量化 RKNN 模型。"""
    dataset_path = make_dataset(spec, output_dir, dataset_limit)
    output_path = output_dir / f"{spec.name}_rk3568.rknn"
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],            # 将 [0,255] 归一化到 [0,1]
            target_platform="rk3568",
            quantized_dtype="asymmetric_quantized-8", # INT8 非对称量化
            quantized_algorithm="normal",
            quantized_method="channel",               # 逐通道量化，精度优于逐层量化
        )
        if ret != 0:
            raise RuntimeError(f"{spec.name}: rknn.config failed: {ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"{spec.name}: rknn.load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        if ret != 0:
            raise RuntimeError(f"{spec.name}: rknn.build failed: {ret}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"{spec.name}: rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()  # 必须释放 RKNN SDK 内部资源

    print(f"[RKNN] {spec.name}: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov5-root", type=pathlib.Path, required=True,
                        help="YOLOv5 源码根目录（用于加载模型类定义）")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "model/rk3568")
    parser.add_argument("--models", nargs="*", choices=[spec.name for spec in MODEL_SPECS],
                        help="指定要导出的模型列表，为空则导出全部")
    parser.add_argument("--onnx-only", action="store_true",
                        help="仅导出 ONNX，不进行 RKNN 转换")
    parser.add_argument("--dataset-limit", type=int, default=50,
                        help="校准数据集图像数量上限（默认 50 张）")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = [spec for spec in MODEL_SPECS if not args.models or spec.name in args.models]
    for spec in selected:
        onnx_path = export_onnx(spec, args.yolov5_root.resolve(), args.output_dir.resolve())
        if not args.onnx_only:
            export_rknn(spec, onnx_path, args.output_dir.resolve(), args.dataset_limit)


if __name__ == "__main__":
    main()
