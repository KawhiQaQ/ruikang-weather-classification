# 睿抗天气图像四分类方案

本仓库整理了睿抗天气图像分类比赛中最终保留的两套方案：线上最强的 EfficientNet-B1 单模，以及比赛末期使用的双 B1 logits ensemble。任务包含 `cloudy`、`rainy`、`snowy`、`sunny` 四类，评价指标为 macro F1。

## 成绩

| 方案 | 线上 macro F1 | 固定本地测试集 macro F1 | 说明 |
|---|---:|---:|---|
| **B1 ExtPre16 + EMA 单模** | **0.9620** | 0.976991 | 线上最优，推荐默认使用 |
| 0.85 ExtPre16 + 0.15 FT-SAM ensemble | 0.9605 | **0.978203** | 最终保留的 ensemble，线上未超过单模 |

线上分数来自比赛平台；固定本地测试集为从 4,999 张训练图像中预留的 10%，只用于最终比较。两种分数不能直接横向等同。

## 核心方法

1. 使用 ImageNet 预训练的 `torchvision.models.efficientnet_b1`，输入尺寸为 240。
2. 从 Kaggle [5-class Weather Status Image Classification](https://www.kaggle.com/datasets/ammaralfaifi/5class-weather-status-image-classification) 中保留 cloudy、rainy、snowy、sunny 四类，共 16,778 张图像，先进行 16 epoch 外部预训练。
3. 比赛训练集固定划分为 80% train、10% validation、10% local test；仅根据 validation macro F1 保存最佳 epoch。
4. 使用轻量增强：`RandomResizedCrop(scale=(0.85, 1.0))`、水平翻转和轻度 ColorJitter。
5. 使用逆频率类别权重、CrossEntropy、`label_smoothing=0.05`、AdamW、CosineAnnealingLR 和 `EMA decay=0.995`。
6. 第二个 ensemble 成员仅在比赛数据微调阶段改用 SAM，`rho=0.05`；最终对两个模型的 logits 按 0.85 / 0.15 加权。

实践中，外部四分类预训练带来的提升最稳定。更大的 backbone、更强数据增强、TTA、额外外部数据、SelfKD 和复杂 stacking 均未在线上超过该 B1 单模。

## 仓库结构

```text
.
├── main.py                       # 双模型 ensemble，平台 predict(X) 入口
├── main_single.py                # 最强单模，平台 predict(X) 入口
├── make_fixed_splits.py          # 固定 80/10/10 分层划分
├── pretrain_external_weather.py  # 外部天气数据预训练
├── train_fixed_split.py          # 普通 B1 + EMA 微调
├── train_sam_fixed_split.py      # SAM + EMA 微调
├── weights/
│   ├── B1_ExtPre16_EMA_repeat3_testf1_0.976991.pth
│   └── B1_ExtPre16EMA_FTSAM_repeat1_testf1_0.978412.pth
└── docs/cv5_results.md           # 5-Fold CV 明细
```

数据集、训练日志、中间 checkpoint 和低收益实验未上传。

## 环境

推荐 Python 3.9 或 3.10：

```bash
conda create -n ruikang-weather python=3.10 -y
conda activate ruikang-weather
pip install -r requirements.txt
```

准备如下目录：

```text
datasets/
├── competition/train/
│   ├── cloudy/
│   ├── rainy/
│   ├── snowy/
│   └── sunny/
└── external/weather_status_4class/
    ├── cloudy/
    ├── rainy/
    ├── snowy/
    └── sunny/
```

仓库不分发比赛数据与外部数据；请分别遵守原比赛和外部数据集的许可条款。

## 复现训练

生成固定划分：

```bash
python make_fixed_splits.py \
  --train-dir datasets/competition/train \
  --output-dir splits \
  --seed 42 \
  --test-ratio 0.10 \
  --val-ratio 0.10
```

进行 16 epoch 外部预训练：

```bash
python pretrain_external_weather.py \
  --backbone efficientnet_b1 \
  --data-dir datasets/external/weather_status_4class \
  --split-file splits/external_weather_4class_seed42_val010.npz \
  --epochs 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --ema-decay 0.995 \
  --seed 42 \
  --output results/pretrain/effb1_external_weather_pretrain_e16.pth
```

在比赛数据上微调单模：

```bash
python train_fixed_split.py \
  --backbone efficientnet_b1 \
  --train-dir datasets/competition/train \
  --splits-dir splits \
  --init-checkpoint results/pretrain/effb1_external_weather_pretrain_e16.pth \
  --epochs 12 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --rrc-scale-min 0.85 \
  --class-weight-power 1.0 \
  --loss-mode ce \
  --ema-decay 0.995 \
  --seed 42 \
  --output results/B1_ExtPre16_EMA.pth
```

训练 SAM 分支：

```bash
python train_sam_fixed_split.py \
  --backbone efficientnet_b1 \
  --train-dir datasets/competition/train \
  --splits-dir splits \
  --init-checkpoint results/pretrain/effb1_external_weather_pretrain_e16.pth \
  --epochs 12 \
  --sam-rho 0.05 \
  --ema-decay 0.995 \
  --seed 42 \
  --output results/B1_ExtPre16EMA_FTSAM.pth
```

同一配置在不同硬件和 PyTorch 版本下仍可能有轻微随机波动。仓库中的最强单模来自固定 `seed=42` 的第 3 次微调，按 validation macro F1 选择最佳 epoch。

## 推理与比赛提交

本地调用 ensemble：

```python
import cv2
import main

image = cv2.imread("example.jpg")
print(main.predict(image))
```

`main.py` 默认加载两份权重并执行 0.85 / 0.15 logits ensemble；`main_single.py` 只加载线上最强单模。两者都接收 OpenCV BGR 格式的 `numpy.ndarray`，返回四个英文类别名之一。

若平台要求入口文件名必须为 `main.py`：

- ensemble：上传当前 `main.py` 和 `weights/` 中两份权重；
- 单模：将 `main_single.py` 作为平台的 `main.py`，只需上传 `B1_ExtPre16_EMA_repeat3_testf1_0.976991.pth`。

## 5-Fold CV

比赛结束后，在全部 4,999 张训练图像上补做 Stratified 5-Fold CV。最强方案 `ExtPre16 + EMA` 的 OOF macro F1 为 **0.971914**，Fold 均值为 **0.971935 ± 0.005706**，同样优于 SelfKD、FT-SAM 和额外外部数据方案。完整结果见 [docs/cv5_results.md](docs/cv5_results.md)。

## 权重校验

```text
19864fb0569706a979eb574580b0b9e430b674caac0eb6838569f592b486b183  B1_ExtPre16_EMA_repeat3_testf1_0.976991.pth
7fe4415b0a9bb9bee7aad6beb55666f84ca2fc258828a91e8bd327f6f33d043e  B1_ExtPre16EMA_FTSAM_repeat1_testf1_0.978412.pth
```

代码采用 MIT License。数据集版权与许可归各自原作者和比赛平台所有。
