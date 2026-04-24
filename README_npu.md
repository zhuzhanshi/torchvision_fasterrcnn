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

## 8. 离线权重使用指南（TorchVision）

在 NPU 环境里，很多机器默认不能联网。  
而 torchvision 的 `weights="DEFAULT"` 首次使用时通常会自动下载权重，所以离线场景建议使用本地权重文件。

### 8.1 为什么需要离线权重

- 训练机/推理机不能访问公网
- 容器网络策略限制外网下载
- 希望固定权重版本，保证可复现

### 8.2 如何下载 torchvision 权重（官方方式）

#### 方式 A：Python 自动下载（推荐）

在可联网机器执行：

```bash
python - <<EOF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
print("downloaded and cached")
EOF
```

说明：
- 首次运行会自动下载
- 缓存目录通常在：
  - `~/.cache/torch/hub/checkpoints/`

#### 方式 B：手动下载（CDN）

torchvision 预训练权重来自 PyTorch 官方 CDN，可手动下载后拷贝到目标机器。  
示例（resnet50_fpn）：
- `https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`

### 8.3 如何放置离线权重

推荐方式 1（项目内管理）：

```text
project/
├── weights/
│   └── fasterrcnn_resnet50_fpn_coco.pth
```

方式 2（任意绝对路径）：
- `/data/weights/fasterrcnn_resnet50_fpn_coco.pth`

### 8.4 如何在 config 中使用

```python
CFG["MODEL"]["WEIGHTS"] = "/path/to/xxx.pth"
CFG["MODEL"]["PRETRAINED"] = False
```

说明：
- 一旦 `MODEL.WEIGHTS` 是本地路径，builder 会优先走本地加载，不再使用 `DEFAULT` 在线权重。
- 该情况下 `PRETRAINED` 会被本地路径逻辑覆盖（等价于被忽略）。

### 8.5 如何验证加载成功

启动训练/测试后，关注日志或 warning：
- 本地权重加载触发后会尝试 `load_state_dict`
- 如果结构有差异，会看到 `missing_keys` / `unexpected_keys` 信息

### 8.6 常见错误排查

错误 1：
- `FileNotFoundError: ...xxx.pth`
- 原因：权重路径写错或文件未挂载进容器

错误 2：
- `missing_keys` / `unexpected_keys`
- 原因：权重文件和当前模型结构（类别数、head 结构、模型变体）不匹配

错误 3：
- `AttributeError: DEFAULT`
- 旧版本 builder 可能因 weights enum 解析方式不稳触发；当前版本已改为显式 registry 解析。

---

## 9. 常见问题（NPU）

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
