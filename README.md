# Torchvision Faster R-CNN Training Framework

一个基于 **PyTorch + torchvision** 的工程化目标检测项目模板，支持 VOC/COCO、自定义训练、测试与推理。

## 目录结构（结构相关）

- `main.py`: 统一入口（参数解析 + mode 调度 + 配置覆盖）
- `train.py`, `test.py`, `infer.py`: 轻量入口封装（固定默认 mode）
- `configs/`: Python 配置系统（`base.py` + 模型配置）
- `engine/runner.py`: 运行编排（构建 runtime / train / test / infer）

## 安装

```bash
pip install -r requirements.txt
```

## 运行入口

### 方式 A：统一入口（推荐）

```bash
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode train --data-root /path/to/data
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode test  --data-root /path/to/data --weights /path/to/model_weights.pth
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode infer --weights /path/to/model_weights.pth --input-path /path/to/image_or_dir
```

### 方式 B：薄封装入口

```bash
python train.py --config configs/fasterrcnn_resnet50_fpn.py --data-root /path/to/data
python test.py  --config configs/fasterrcnn_resnet50_fpn.py --data-root /path/to/data --weights /path/to/model_weights.pth
python infer.py --config configs/fasterrcnn_resnet50_fpn.py --weights /path/to/model_weights.pth --input-path /path/to/image_or_dir
```

## 常用 CLI 覆盖项

- `--mode {train,test,infer}`
- `--data-root`
- `--output-root`
- `--exp-name`
- `--batch-size`
- `--epochs`
- `--lr`
- `--resume`（恢复完整训练状态）
- `--weights`（仅加载模型权重）
- `--device`
- `--input-path`
- `--save-vis {true|false}`
- `--num-workers`
- `--amp {true|false}`

> 兼容旧写法：以上参数也接受下划线形式（如 `--data_root`）。

## 配置系统说明

`configs/base.py` 是单一真源，包含以下一级块：

- `RUNTIME`
- `DATASET`
- `INPUT`
- `AUG`
- `DATALOADER`
- `MODEL`
- `OPTIMIZER`
- `SCHEDULER`
- `TRAIN`
- `EVAL`
- `INFER`
- `LOG`

### 支持的 Faster R-CNN 结构（`MODEL.NAME`）

- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `fasterrcnn_mobilenet_v3_large_fpn`
- `fasterrcnn_mobilenet_v3_large_320_fpn`

可通过 `--config configs/<model_preset>.py` 切换。

### NUM_CLASSES 约定

- `DATASET.NUM_CLASSES`：前景类别数（不含背景）
- `MODEL.NUM_CLASSES`：前景类别数（不含背景）
- 背景类由模型构建器内部自动 `+1` 处理。

### RESUME 与 WEIGHTS 的语义

- `RESUME`：恢复完整训练状态（model + optimizer + scheduler + scaler + epoch）
- `WEIGHTS`：仅加载模型参数（微调/测试/推理，不恢复 optimizer/scheduler/scaler）

> 建议：训练中断续训使用 `RESUME`；评估/推理或迁移学习初始化使用 `WEIGHTS`。

## Config Snapshot

- 默认会在每次运行时把“最终生效配置（含 CLI 覆盖）”写入：

```text
outputs/{model_name}/{exp_name}/{timestamp}/config_snapshot.py
```

- 可通过 `LOG.SAVE_CONFIG_SNAPSHOT` 控制开关。

## 输出目录

```text
outputs/{model_name}/{exp_name}/{timestamp}/
├── config_snapshot.py
├── train.log
├── tb/
├── checkpoints/
├── eval/
└── infer/
```
