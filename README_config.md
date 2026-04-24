# README_config.md

本指南面向“使用者”，目标是：  
**快速修改配置，控制训练行为，不需要了解内部实现细节。**

---

## 1. 配置入口

主配置在：

- `configs/base.py`（默认配置）
- `configs/fasterrcnn_*.py`（模型预设）

推荐方式：从某个模型预设开始，然后用 CLI 覆盖关键参数。

---

## 2. 最常用配置（先看这几块）

## 2.1 DATASET

```python
"DATASET": {
    "TYPE": "voc",          # voc | coco
    "ROOT": "data",
    "TRAIN_SPLIT": "train",
    "VAL_SPLIT": "val",
    "TEST_SPLIT": "test",
    "CLASSES": ["class1"],  # 前景类，不含背景
    "NUM_CLASSES": 1,       # 必须等于 len(CLASSES)
}
```

常用增强/过滤：
- `FILTER_EMPTY_GT`
- `MIN_BOX_AREA`
- `IGNORE_DIFFICULT`（VOC）
- `CHECK_DATASET`

---

## 2.2 MODEL

```python
"MODEL": {
    "NAME": "fasterrcnn_resnet50_fpn",
    "NUM_CLASSES": 1,         # 前景类数量，不含背景
    "REPLACE_HEAD": True,
}
```

支持模型：
- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `fasterrcnn_mobilenet_v3_large_fpn`
- `fasterrcnn_mobilenet_v3_large_320_fpn`

---

## 2.3 TRAIN / OPTIMIZER

```python
"TRAIN": {
    "EPOCHS": 12,
    "ACCUMULATION_STEPS": 1,
    "VALIDATE_EVERY_EPOCH": 1,
}
"OPTIMIZER": {
    "NAME": "sgd",   # sgd | adamw
    "LR": 0.005,
}
```

---

## 2.4 RUNTIME（运行控制）

```python
"RUNTIME": {
    "MODE": "train",     # train | test | infer
    "DEVICE": "cuda",    # cpu | cuda | npu
    "WEIGHTS": "",       # 加载模型权重（模型参数）
    "RESUME": "",        # 恢复训练状态（模型+优化器+调度器+epoch）
}
```

---

## 3. 最小可运行示例

> 假设你已准备好 VOC 数据目录 `/path/to/voc_root`

```bash
python main.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --mode train \
  --data-root /path/to/voc_root \
  --device cuda \
  --epochs 2 \
  --batch-size 2 \
  --lr 0.001
```

---

## 4. CLI 覆盖（最实用）

常用可直接覆盖：

- `--data-root`
- `--output-root`
- `--exp-name`
- `--batch-size`
- `--epochs`
- `--lr`
- `--resume`
- `--weights`
- `--device`
- `--input-path`
- `--save-vis`
- `--num-workers`
- `--amp`
- `--local-rank`（分布式时由 torchrun 注入）

示例：

```bash
python train.py \
  --config configs/fasterrcnn_mobilenet_v3_large_fpn.py \
  --data-root /path/to/data \
  --batch-size 4 \
  --epochs 20 \
  --lr 0.002 \
  --device npu
```

---

## 5. NUM_CLASSES 约定（非常重要）

- `DATASET.CLASSES`：前景类别名列表（不含背景）
- `DATASET.NUM_CLASSES == len(DATASET.CLASSES)`（必须相等）
- `MODEL.NUM_CLASSES` 也应与其一致（系统会对齐并校验）
- 背景类由模型内部处理（label=0），不要写进 `CLASSES`

---

## 6. WEIGHTS vs RESUME（区别）

- `WEIGHTS`：只加载模型参数  
  - 用于：测试、推理、迁移学习初始化
- `RESUME`：恢复完整训练状态  
  - 包括模型、优化器、调度器、AMP scaler、epoch、best metric

一句话：
- **继续训练中断任务** -> `RESUME`
- **只想加载已有模型进行 test/infer** -> `WEIGHTS`

---

## 7. config snapshot

每次运行会把“最终生效配置（含 CLI 覆盖）”保存到：

```text
outputs/{model}/{exp}/{timestamp}/config_snapshot.py
```

这有助于复现实验。

---

## 8. 输入尺寸与增强配置说明（Faster R-CNN 重点）

### 8.1 Faster R-CNN 不是固定输入尺寸

`torchvision` 的 Faster R-CNN 默认是“短边缩放 + 长边约束”策略，不是固定 `HxW`。  
你主要控制的是：

- `MODEL.MIN_SIZE`
- `MODEL.MAX_SIZE`

训练时，模型内部 transform 会把图片按这两个参数做缩放（并保持比例）。

### 8.2 random resize 的作用

`AUG.TRAIN.RANDOM_RESIZE` 会在训练数据层随机选择短边尺度，提升尺度鲁棒性。  
这和模型内部 `MIN_SIZE/MAX_SIZE` 是两层机制：  
- 数据增强层：随机扰动  
- 模型层：最终进入检测器前的约束缩放

### 8.3 random crop 的风险

检测任务里随机裁剪容易带来：
- 目标框被大量裁掉
- 小目标完全消失
- 类别分布偏移

如果数据本身目标较小或稀疏，建议先关闭或谨慎调小裁剪强度。

### 8.4 eval / infer 如何设置图片尺寸

你可以单独覆盖验证/推理阶段的模型 transform：

- `EVAL.MIN_SIZE` / `EVAL.MAX_SIZE`
- `INFER.MIN_SIZE` / `INFER.MAX_SIZE`

当这些值 > 0 时，会在 eval/infer 阶段覆盖模型默认 transform 尺寸。  
若为 0，则沿用模型默认配置（`MODEL.MIN_SIZE/MAX_SIZE`）。

---

## 9. 数据集统计（训练前）

`DATASET` 新增字段：

- `STATS_BEFORE_TRAIN`: 训练开始前是否统计
- `SAVE_STATS`: 是否保存统计文件
- `STATS_SPLITS`: 要统计的 split 列表（如 `["train", "val", "test"]`）
- `STATS_BBOX_DISTRIBUTION`: 是否统计 bbox 尺度分布（默认 `False`）

统计输出目录：

```text
outputs/{model}/{exp}/{timestamp}/dataset_stats/
├── dataset_stats.json
├── dataset_stats.csv
└── dataset_stats.txt
```

包含：
- 各 split 图片数量
- 各类别框数量（box_count）
- 各类别出现图片数（image_count）
- 空标注图片数量
- bbox 尺度分布（small / medium / large，只有 `STATS_BBOX_DISTRIBUTION=True` 时启用）

说明：
- 默认 `STATS_BBOX_DISTRIBUTION=False`，速度更快（适合常规训练）
- 当你需要分析目标尺度分布、校验 anchor 设计时，再开启 `True`

---

## 10. 验证调度策略（DURING / AFTER）

`EVAL` 推荐关键字段：

```python
"EVAL": {
    "DURING_TRAIN": True,
    "INTERVAL": 1,
    "AFTER_TRAIN": True,
    "BEST_METRIC": "map",
}
```

支持四种策略：

1) 训练中周期验证  
- `DURING_TRAIN=True`, `INTERVAL=1`

2) 只在训练结束后验证一次  
- `DURING_TRAIN=False`, `AFTER_TRAIN=True`

3) 两者都启用  
- 训练中按 interval 验证 + 训练结束后 final eval

4) 两者都关闭  
- `DURING_TRAIN=False`, `AFTER_TRAIN=False`

注意：
- `best checkpoint` 只基于训练过程中的验证结果更新
- 训练后 final eval 结果会保存到 `eval/final/final_metrics.json`

---

## 11. 常见错误与排查

### 错误1：NUM_CLASSES 不一致
症状：启动时报类别数相关错误。  
排查：确保：
- `len(DATASET.CLASSES) == DATASET.NUM_CLASSES`
- `MODEL.NUM_CLASSES` 与上面一致

### 错误2：找不到数据集 split
症状：提示 split 文件或标注文件不存在。  
排查：
- VOC：`ImageSets/Main/{split}.txt` 是否存在
- COCO：`annotations/instances_{split}.json` 是否存在

### 错误3：把 RESUME 当成 WEIGHTS 用
症状：test/infer 时加载失败或行为异常。  
排查：
- test/infer 只用 `--weights`
- 训练续跑才用 `--resume`

### 错误4：device 设置与环境不匹配
症状：`cuda`/`npu` 不可用报错。  
排查：
- 改用 `--device cpu` 先验证流程
- NPU 环境需安装匹配版本 `torch_npu`

### 错误5：多卡训练日志重复
本项目已做 rank0 写入控制；如仍重复，请确认是否重复启动了多个任务组。
