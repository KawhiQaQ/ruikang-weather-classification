# 2026睿抗智海-天气分类

本仓库整理了睿抗天气图像分类比赛中最终保留的两套方案：长期最强的 EfficientNet-B1 单模，以及比赛最终提交的双 B1 logits ensemble。任务包含 `cloudy`、`rainy`、`snowy`、`sunny` 四类，评价指标为 macro F1。

## 成绩

| 方案 | 线上 macro F1 | 固定本地测试集 macro F1 | 说明 |
|---|---:|---:|---|
| B1 ExtPre16 + EMA 单模 | 0.9620 | 0.976991 | 长期最强单模 |
| **0.85 ExtPre16 + 0.15 FT-SAM ensemble** | **0.9641** | **0.978203** | 最终提交，线上最高 |

线上分数来自比赛平台。固定本地测试集为从 4,999 张训练图像中预留的 10%，只用于最终比较；线上与本地分数不能直接横向等同。

## 核心方法

1. Backbone 为 **EfficientNet-B1**（`torchvision.models.efficientnet_b1`），单模和 ensemble 的两个成员均使用该架构，输入尺寸为 240。
2. 从 Kaggle [5-class Weather Status Image Classification](https://www.kaggle.com/datasets/ammaralfaifi/5class-weather-status-image-classification) 中保留 cloudy、rainy、snowy、sunny 四类，共 16,778 张图像，先进行 16 epoch 外部预训练。
3. 比赛训练集固定划分为 80% train、10% validation、10% local test；仅根据 validation macro F1 保存最佳 epoch。
4. 使用轻量增强：`RandomResizedCrop(scale=(0.85, 1.0))`、水平翻转和轻度 ColorJitter。
5. 使用逆频率类别权重、CrossEntropy、`label_smoothing=0.05`、AdamW、CosineAnnealingLR 和 `EMA decay=0.995`。
6. 第二个 ensemble 成员仅在比赛数据微调阶段改用 SAM，`rho=0.05`；最终对两个模型的 logits 按 0.85 / 0.15 加权。

实践中，外部四分类预训练带来的提升最稳定。更大的 backbone、更强数据增强、TTA、额外外部数据、SelfKD 和复杂 stacking 均未形成更可靠的单模型；最终仅用小权重 FT-SAM 分支与主模型做保守集成。

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

## 推理

本地调用 ensemble：

```python
import cv2
import main

image = cv2.imread("example.jpg")
print(main.predict(image))
```

`main.py` 默认加载两份权重并执行 0.85 / 0.15 logits ensemble；`main_single.py` 只加载长期最强单模。两者都接收 OpenCV BGR 格式的 `numpy.ndarray`，返回四个英文类别名之一。

## 失败尝试与复盘

- **更大或更新的 Backbone**

  尝试内容：测试了 EfficientNet-B2/B3/B7、ResNet50、ConvNeXt/ConvNeXtV2、Swin-T、DeiT-S 和 DINOv3 等架构，均未稳定超过 EfficientNet-B1。

  可能原因：比赛训练集只有 4,999 张图像，更大的模型方差更高、也更容易过拟合；当前外部数据规模和训练超参数与 B1 的容量更匹配，预训练模型更大并不等于目标域效果更好。

- **更强的数据增强**

  尝试内容：将 `RandomResizedCrop` 扩大到 `(0.75, 1.0)`，增强 ColorJitter，并加入 RandomGrayscale 和 RandomErasing，5-Fold 分数由原版约 0.9335 降至约 0.9316。

  可能原因：天气分类依赖整幅图像的色彩、亮度和云层纹理，过强裁剪与颜色扰动会破坏这些类别线索；增强强度应贴近真实拍摄变化，而不是越强越好。

- **激进的类别平衡策略**

  尝试内容：使用 `WeightedRandomSampler` 并移除 loss class weight，三次结果约为 0.9454、0.9345、0.9377；Balanced Softmax 也出现少数类 recall 上升、precision 下降的现象。

  可能原因：类别不平衡并未严重到需要重采样，重复少数类样本容易同时放大噪声并改变真实类别先验。最终采用普通 shuffle 配合逆频率加权 CE 更稳定。

- **TTA 推理增强**

  尝试内容：测试了水平翻转、几何裁剪，以及亮度抖动、高斯噪声和条纹干扰等多视图 TTA，大部分设置使验证或线上分数下降，同时成倍增加 CPU 推理开销。

  可能原因：部分变换并非严格保持天气语义，平均 logits 会把有偏预测一起混入；同一模型不同视图的错误相关性也很高，无法保证通过平均获得有效互补。

- **Self-Distillation 与 PatchSwap**

  尝试内容：测试了 SelfKD、ensemble teacher 蒸馏和同类别 PatchSwap。部分方案本地达到 0.977 至 0.979，但 SelfKD、EnsembleKD 和 PatchSwap 的线上分数分别仅为 0.9507、0.9519 和 0.9450。

  可能原因：教师和学生均为同结构 B1，错误高度相关；小规模固定划分容易高估蒸馏收益。PatchSwap 还可能破坏天气场景所需的全局空间与纹理一致性。

- **扩展外部数据与域适配**

  尝试内容：增加更多外部天气数据后，本地单次最高达到 0.981818，但线上反而下降；数据过滤、AdaBN 和 source data soft weighting 也没有稳定提升。

  可能原因：新增数据与线上目标域在场景、类别定义和图像质量上存在偏移，更多数据可能引入额外噪声。依据当前模型过滤源数据还会强化模型已有偏差，造成对本地划分的虚假提升。

- **复杂 Ensemble 与 Stacking**

  尝试内容：搜索模型权重、temperature 和 class bias 后，validation macro F1 可达 0.978888，但 local test 仅为 0.977014；GBDT stacking 更出现 validation 1.0、local test 0.954758 的严重过拟合。FT-SAM 单模线上为 0.9577，也低于主模型的 0.9620。

  可能原因：仅用 500 张 validation 图像学习融合参数，自由度远高于有效样本量；多个 B1 分支的错误又高度相关。最终固定使用 `0.85 ExtPre16 + 0.15 FT-SAM`，让较弱分支只提供少量互补信息，线上提升至 0.9641。

- **同架构 5-Fold 推理集成**

  尝试内容：在固定 local test 之外训练 5 个早期 B1 fold 模型并平均 logits，local test macro F1 为 0.946245，低于同阶段单模的 0.953633。

  可能原因：每个 fold 只能使用部分开发数据，且相同架构、初始化与训练策略产生的错误高度相关。5-Fold 更适合评估方案的稳定性，并不保证将 5 个模型用于推理就一定提分。

## 5-Fold CV

比赛结束后，在全部 4,999 张训练图像上补做 Stratified 5-Fold CV。最强方案 `ExtPre16 + EMA` 的 OOF macro F1 为 **0.971914**，Fold 均值为 **0.971935 ± 0.005706**，同样优于 SelfKD、FT-SAM 和额外外部数据方案。完整结果见 [docs/cv5_results.md](docs/cv5_results.md)。

## 许可证

代码采用 MIT License。数据集版权与许可归各自原作者和比赛平台所有。
