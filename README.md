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

可选依赖说明：
- `pycocotools`：COCO 数据集读取与 COCO 风格评估必须。
- `tensorboard`：仅在 `LOG.TENSORBOARD=true` 时需要；若缺失会自动降级并打印 warning。
- `torch_npu`：仅 Ascend NPU 运行需要。由于版本与 CANN/torch 强相关，不在 `requirements.txt` 强制安装。

## 运行入口

### 方式 A：统一入口（推荐）

```bash
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode train --data-root /path/to/data
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode test  --data-root /path/to/data --weights /path/to/model_weights.pth
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode infer --weights /path/to/model_weights.pth --input-path /path/to/image_or_dir
```

### Ascend NPU（单卡）示例

```bash
# train
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode train --device npu --data-root /path/to/data

# test
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode test --device npu --weights /path/to/model_weights.pth --data-root /path/to/data

# infer
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode infer --device npu --weights /path/to/model_weights.pth --input-path /path/to/image_or_dir
```

说明：
- 当前实现为**单卡 NPU 支持**（不含 DDP/HCCL）。
- 当 `RUNTIME.DEVICE=npu` 时会在运行时尝试导入 `torch_npu`；若缺失会给出明确错误。

### 方式 B：薄封装入口

```bash
python train.py --config configs/fasterrcnn_resnet50_fpn.py --data-root /path/to/data
python test.py  --config configs/fasterrcnn_resnet50_fpn.py --data-root /path/to/data --weights /path/to/model_weights.pth
python infer.py --config configs/fasterrcnn_resnet50_fpn.py --weights /path/to/model_weights.pth --input-path /path/to/image_or_dir
```

### TensorBoard 启动

```bash
tensorboard --logdir outputs
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

## 数据集组织与类别约定（VOC/COCO）

- `DATASET.CLASSES`：前景类别名列表（不包含背景）。
- `DATASET.NUM_CLASSES`：前景类别数量，必须与 `len(CLASSES)` 一致。
- 训练内部标签约定：`background=0`，前景类从 `1` 开始；VOC/COCO 都会 remap 到这一统一体系。
- `DATASET.TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT` 控制三套数据集构建，`datasets/builder.py` 会统一读取。

### VOC

```text
ROOT/
├── Annotations/
├── JPEGImages/
└── ImageSets/Main/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### COCO

```text
ROOT/
├── train2017/ (or train)
├── val2017/   (or val)
├── test2017/  (or test)
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── instances_test2017.json
```

可通过 `TRAIN_SPLIT/VAL_SPLIT/TEST_SPLIT` 改成 `train2017/val2017/test2017` 等真实目录名。

### 常用数据过滤配置

- `DATASET.FILTER_EMPTY_GT`：训练集过滤空标注图像
- `DATASET.MIN_BOX_AREA`：过滤过小框
- `DATASET.IGNORE_DIFFICULT`：VOC 中忽略 difficult 目标
- `DATASET.CHECK_DATASET`：启用更严格的文件存在性检查

## 预处理与增强（Detection）

- 所有 transforms 都使用 detection 接口：`image, target -> image, target`。
- `target["boxes"]` 会随几何增强同步更新（flip / resize / crop）。
- 颜色增强（如 `ColorJitter`）只改像素，不改 bbox。
- train 与 val/test 口径分离：
  - train：`AUG.TRAIN.*`（`HFLIP/VFLIP/COLOR_JITTER/RANDOM_RESIZE/RANDOM_CROP`）
  - val/test：`AUG.TEST.*`（可选 resize）
- `datasets/transforms.py` 仅消费 `INPUT.*` 作为数据侧预处理配置；模型 transform 参数由 `MODEL.*` 在 model builder 侧消费。

### Detection Collate 约定

- DataLoader 使用 `datasets/collate.py` 的 `detection_collate_fn`。
- batch 组织形式为：
  - `images: list[Tensor[C,H,W]]`
  - `targets: list[dict]`
- 不会将不同尺寸图像强行 `stack` 成单个 tensor（符合 torchvision detection 预期）。

### RESUME 与 WEIGHTS 的语义

- `RESUME`：恢复完整训练状态（model + optimizer + scheduler + scaler + epoch）
- `WEIGHTS`：仅加载模型参数（微调/测试/推理，不恢复 optimizer/scheduler/scaler）

> 建议：训练中断续训使用 `RESUME`；评估/推理或迁移学习初始化使用 `WEIGHTS`。

## 训练链路说明（Trainer）

- 训练主流程集中在 `engine/trainer.py`：`train()` / `train_one_epoch()` / `validate()` / `save_checkpoint()`。
- 支持 detection loss dict 汇总（`loss_classifier/loss_box_reg/loss_objectness/loss_rpn_box_reg` + `loss_total`）。
- 支持 AMP（`RUNTIME.USE_AMP` 或 `--amp`）、梯度累积（`TRAIN.ACCUMULATION_STEPS`）、梯度裁剪（`TRAIN.GRAD_CLIP`）。
- scheduler 默认按 **epoch** `step`（支持 `StepLR/MultiStepLR/CosineAnnealingLR`）。
- checkpoint 默认保存到 `checkpoints/latest.pth`，best 指标更新时保存 `checkpoints/best.pth`，并包含：
  - `model/optimizer/scheduler/scaler`
  - `epoch`
  - `best_metric`
  - 关键配置摘要

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
├── env.txt
├── train.log
├── events.jsonl (optional, LOG.JSON=true)
├── tb/
├── checkpoints/
├── eval/
└── infer/
```

- 每次运行默认都会创建新的 `timestamp` 目录，避免覆盖历史实验。
- 如需复用现有实验目录（常用于 test/infer 复用同一目录），可设置 `RUNTIME.EXISTING_OUTPUT_DIR`。

## 验证 / 测试评估口径（Evaluator）

- 评估统一由 `engine/evaluator.py` 的 `Evaluator` 执行：
  - `test.py` 仅是薄入口，实际评估由 `main.py -> engine/runner.py -> Evaluator` 调用。
  - `trainer.validate()` 与 `test` 模式复用同一套 `Evaluator.evaluate()` 逻辑。
- 当前默认评估指标为 `EVAL.METRIC="bbox"`，基于 COCO 风格统计：
  - `map`（mAP@[0.5:0.95]）
  - `map50`
  - `map75`
  - `ar`
  - 可选 `per_class_ap`（`EVAL.PER_CLASS_AP=true`）

### EVAL 关键配置

- `EVAL.ENABLE`：训练期间是否启用周期验证
- `EVAL.INTERVAL`：训练中验证间隔（epoch）
- `EVAL.SCORE_THRESH`：评估阶段预测分数阈值（仅评估链路）
- `EVAL.MAX_DETS`：评估阶段每图最多保留框数
- `EVAL.SAVE_PREDICTIONS`：保存 `predictions.json`
- `EVAL.SAVE_GT`：保存 `ground_truth.json`
- `EVAL.PER_CLASS_AP`：输出 per-class AP
- `EVAL.USE_COCO_EVAL`：是否使用 COCO 风格 bbox 评估（当前要求为 `true`）

> 说明：`EVAL.*` 与 `INFER.*` 分离；评估不会读取 `INFER.SCORE_THRESH / INFER.MAX_DETS`。

### 评估结果保存位置

- **train 周期验证**：`eval/epoch_{xxx}/`
- **test 模式**：`eval/test/`

每次评估至少产出：

- `metrics.json`
- `per_class_ap.csv` / `per_class_ap.json`（启用 per-class AP 时）
- `predictions.json`（`EVAL.SAVE_PREDICTIONS=true`）
- `ground_truth.json`（`EVAL.SAVE_GT=true`）

### 依赖说明

- COCO 风格评估依赖 `pycocotools`（已在 `requirements.txt` 中声明）。

## 推理流程（Inferencer）

- 推理统一由 `engine/inferencer.py` 的 `Inferencer` 执行；`infer.py` 保持薄入口。  
- `runner` 仅负责装配模型/日志/输出目录，推理后处理与结果保存都在 Inferencer 内完成。

### 输入形式

- `INFER.INPUT_PATH` / `--input-path` 支持：
  - 单图路径
  - 文件夹路径
- 文件夹默认非递归；若 `INFER.RECURSIVE=true` 则递归扫描子目录。
- 支持后缀：`.jpg/.jpeg/.png/.bmp/.webp`。

### 推理输出目录

统一保存到：

```text
outputs/{model_name}/{exp_name}/{timestamp}/infer/
├── vis/
├── json/
├── txt/
└── predictions_all.json
```

- `INFER.SAVE_VIS`：是否保存可视化图到 `infer/vis/`
- `INFER.SAVE_JSON`：是否保存每图结构化结果到 `infer/json/`
- `INFER.SAVE_TXT`：是否保存每图文本结果到 `infer/txt/`
- `predictions_all.json`：总汇总（成功/失败计数 + 所有结果）

### 后处理与可视化配置

- `INFER.SCORE_THRESH`：置信度阈值
- `INFER.MAX_DETS`：每图最大输出框数
- `INFER.NMS_THRESH`：覆盖模型 ROI NMS 阈值（infer-time）
- `INFER.CLASS_FILTER`：可按**类别名**或**label id**过滤（前景从 1 开始）
- `INFER.DRAW_LABEL / DRAW_SCORE / LINE_THICKNESS`：可视化标注控制

> 说明：`INFER.*` 与 `EVAL.*` 参数独立，推理不会读取 `EVAL` 阈值配置。

## 日志与实验管理（Logger/TensorBoard）

- 统一日志入口：`utils/logger.py`，业务模块只依赖 `ExperimentLogger`（`info/warning/error/log_scalars`）。
- `LOG.TXT=true` 时写入 `train.log`（同时输出到控制台）。
- `LOG.TENSORBOARD=true` 时写入 `tb/`，可与 txt 独立或同时启用。
- `LOG.JSON=true` 时写入 `events.jsonl`（结构化日志与标量记录）。
- `LOG.SAVE_CONFIG_SNAPSHOT=true` 时保存最终生效配置到 `config_snapshot.py`。
- `LOG.SAVE_ENV_INFO=true` 时保存环境信息到 `env.txt`（Python/torch/torchvision/CUDA/device）。

## 工具脚本（tools）

主流程入口是 `main.py / train.py / test.py / infer.py`；`tools/` 是辅助排查工具。

### 1) 数据集检查：`tools/check_dataset.py`

```bash
python tools/check_dataset.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --split train \
  --max-samples 1000 \
  --output-json outputs/check_dataset_train.json
```

检查项：
- target 必要字段缺失
- 空标注样本
- bbox 宽高非法
- bbox 越界
- label 非法
- 类别计数

### 2) GT 可视化：`tools/vis_gt.py`

```bash
python tools/vis_gt.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --split train \
  --max-samples 50 \
  --output-dir outputs/vis_gt_train \
  --disable-train-aug
```

- 保存可视化图片到 `--output-dir`
- 同时写出 `meta.json` 便于追踪原样本

### 3) 预测导出：`tools/export_predictions.py`

```bash
# infer 聚合结果 -> csv
python tools/export_predictions.py \
  --input outputs/.../infer/predictions_all.json \
  --output outputs/.../infer/predictions.csv \
  --format csv

# eval 预测结果 -> txt
python tools/export_predictions.py \
  --input outputs/.../eval/test/predictions.json \
  --output outputs/.../eval/test/predictions.txt \
  --format txt
```

支持输入：
- `infer/predictions_all.json`
- `infer/json/*.json`（单图结构化结果）
- `eval/*/predictions.json`

## 基础排错建议

- 缺少 `pycocotools`：COCO 数据集和评估不可用，请先安装 requirements。
- 缺少 `tensorboard`：训练仍可运行，但不会写入 `tb/`。
- 工具脚本报输入路径错误：优先检查 `--input-path`/`--input` 是否存在。
- 权重加载失败：确认 `--weights` 文件存在且与类别配置一致。
