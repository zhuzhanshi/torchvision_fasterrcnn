# README_npu.md

本指南面向“使用者”，目标是：  
**在 Ascend NPU 上快速启动本项目（单卡 + 多卡训练）**。

> 说明：
> - 本项目已支持 `DEVICE=npu` 的单卡路径。
> - 本项目已支持 DDP 多卡训练（NPU backend=HCCL）。
> - infer 当前按单机单设备使用（不做分布式 infer）。

---

## 1. 环境准备（NPU）

你需要一套彼此兼容的版本组合：

- CANN / 驱动 / 固件
- PyTorch
- torchvision
- torch_npu（Ascend 版本）

推荐做法：先按 Ascend 官方兼容矩阵安装 `torch + torchvision + torch_npu`，再安装本项目依赖：

```bash
pip install -r requirements.txt
```

> `requirements.txt` 不强制安装 `torch_npu`，因为它与 CANN/PyTorch 版本强耦合，需要你按机器环境单独安装。

---

## 2. 单卡 NPU 训练

### 2.1 命令（推荐统一入口）

```bash
python main.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --mode train \
  --device npu \
  --data-root /path/to/data
```

### 2.2 也可用薄入口

```bash
python train.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --device npu \
  --data-root /path/to/data
```

---

## 3. 单卡 NPU 测试与推理

### 3.1 test（评估）

```bash
python main.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --mode test \
  --device npu \
  --weights /path/to/model_weights.pth \
  --data-root /path/to/data
```

### 3.2 infer（单图或目录）

```bash
python main.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --mode infer \
  --device npu \
  --weights /path/to/model_weights.pth \
  --input-path /path/to/image_or_dir
```

> infer 继续按单机单设备运行，不做分布式 infer。

---

## 4. 多卡 NPU 训练（DDP + HCCL）

使用 `torchrun` 启动（示例 2 卡）：

```bash
torchrun --nproc_per_node=2 main.py \
  --config configs/fasterrcnn_resnet50_fpn.py \
  --mode train \
  --device npu \
  --data-root /path/to/data
```

项目会自动：
- 读取 `WORLD_SIZE/RANK/LOCAL_RANK`（torchrun 环境变量）
- 选择 NPU 分布式 backend：`hccl`
- 使用 `DistributedDataParallel`
- 训练集使用 `DistributedSampler`

---

## 5. 关键配置项（NPU + DDP）

在 `configs/base.py` 的 `RUNTIME` 中：

- `DEVICE`: `"cpu" | "cuda" | "npu"`
- `DISTRIBUTED`: 是否启用分布式（torchrun 下会自动接入）
- `DIST_URL`: 默认 `env://`
- `DIST_BACKEND`: 可留空，程序会按设备自动选：
  - npu -> hccl
  - cuda -> nccl
  - cpu -> gloo
- `WORLD_SIZE / RANK / LOCAL_RANK`: torchrun 注入

---

## 6. 训练行为（多卡时你会看到）

- 只有 **rank0** 写全局输出：
  - `train.log`
  - TensorBoard 事件
  - `checkpoints/latest.pth` / `checkpoints/best.pth`
  - `eval` 下最终指标文件
- 其他 rank 不重复写这些全局文件。
- 验证/测试会跨 rank 汇总预测与 GT，再由 rank0 统一计算指标。

---

## 7. 输出目录

默认：

```text
outputs/{model_name}/{exp_name}/{timestamp}/
├── config_snapshot.py
├── env.txt
├── train.log
├── tb/
├── checkpoints/
├── eval/
└── infer/
```

---

## 8. 常见问题（NPU）

### Q1: 报错 “torch_npu is not available”
你把 `--device npu` 打开了，但环境里没有可用 `torch_npu`。  
请按 Ascend 兼容矩阵安装正确版本的 `torch_npu`。

### Q2: 多卡启动卡住 / 无输出
优先检查：
- `torchrun` 是否正确启动
- HCCL 环境变量与网络配置
- NPU 可见卡配置是否正确
- CANN/驱动版本是否匹配

### Q3: AMP 在 NPU 表现与 CUDA 不同
当前实现对 NPU AMP 使用了保守兼容策略：  
- 能 autocast 则用 autocast
- 不可用则回退 FP32  
这是为了优先保证稳定可运行。

### Q4: infer 能否多卡并行？
当前版本不做分布式 infer。建议单机单设备 infer。
