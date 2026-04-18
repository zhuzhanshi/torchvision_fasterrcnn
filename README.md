# Torchvision Faster R-CNN Training Framework

一个基于 **PyTorch + torchvision** 的工程化目标检测项目模板，支持 VOC/COCO、自定义训练、测试与推理。

## 目录结构

- `main.py`: 统一入口（train/test/infer）
- `train.py`, `test.py`, `infer.py`: 薄封装入口
- `configs/`: Python 配置系统
- `models/`: 模型构建与组件
- `datasets/`: 数据集、变换、dataloader
- `engine/`: Trainer / Evaluator / Inferencer
- `utils/`: logger、checkpoint、config、visualize 等
- `tools/`: 数据检查、导出、可视化工具
- `outputs/`: 实验输出目录

## 安装

```bash
pip install -r requirements.txt
```

## 数据组织

### VOC

```text
data_root/
├── Annotations/
├── JPEGImages/
└── ImageSets/Main/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### COCO

```text
data_root/
├── train/
├── val/
├── test/
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

## 训练

```bash
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode train \
  --data_root /path/to/data --exp_name exp1 --batch_size 2 --epochs 12
```

## 测试

```bash
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode test \
  --data_root /path/to/data --weights /path/to/best.pth
```

## 推理

```bash
python main.py --config configs/fasterrcnn_resnet50_fpn.py --mode infer \
  --weights /path/to/best.pth --input_path /path/to/image_or_dir --save_vis true
```

## 配置说明

- `RUNTIME`: 运行参数、设备、输出目录
- `DATASET`: 数据集类型、类别、划分
- `INPUT` / `AUG`: 输入预处理与增强
- `MODEL`: 模型结构、权重、RPN/ROIHeads 参数
- `OPTIMIZER` / `SCHEDULER`: 优化器与学习率策略
- `TRAIN` / `EVAL` / `INFER`: 训练评估推理控制
- `LOG`: txt/tensorboard 日志

## 输出目录

实验输出示例：

```text
outputs/{model_name}/{exp_name}/{timestamp}/
├── config_snapshot.py
├── train.log
├── tb/
├── checkpoints/
├── eval/
└── infer/
```

## 扩展建议

- 在 `models/faster_rcnn.py` 增加新 torchvision detection 模型映射。
- 在 `datasets/transforms.py` 扩展 Mosaic/MixUp 等高级增强（TODO）。
- 在 `engine/evaluator.py` 为 VOC 增加官方风格指标（TODO）。
