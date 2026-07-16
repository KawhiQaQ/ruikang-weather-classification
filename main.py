from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models

try:
    import cv2
except ImportError:
    cv2 = None


label = ["cloudy", "rainy", "snowy", "sunny"]
im_size = 240
mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
ROOT = Path(__file__).resolve().parent
MODEL_ENSEMBLE = [
    ("B1_ExtPre16_EMA_repeat3_testf1_0.976991.pth", 0.85),
    ("B1_ExtPre16EMA_FTSAM_repeat1_testf1_0.978412.pth", 0.15),
]
USE_TTA = False
TTA_CROP_RATIO = 0.90
BACKBONES = {
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b7": models.efficientnet_b7,
    "resnet50": models.resnet50,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


def build_model(num_classes=4, backbone="efficientnet_b1"):
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone in checkpoint: {backbone}")
    factory = BACKBONES[backbone]
    model = factory(weights=None)
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model head for backbone: {backbone}")
    return model


def find_model_path(model_name):
    candidates = []
    search_roots = [
        Path.cwd(),
        Path.cwd() / "weights",
        Path.cwd() / "results",
        ROOT,
        ROOT / "weights",
        ROOT / "results",
        Path("/home/jovyan/work"),
        Path("/home/jovyan/work/results"),
    ]
    for root in search_roots:
        candidates.append(root / model_name)

    for path in candidates:
        if path.exists():
            return path

    return None


def load_single_model(model_name):
    global im_size
    model_path = find_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(
            "Missing model checkpoint: "
            f"{model_name}. Expected it under ./weights, ./results, "
            "or /home/jovyan/work/results."
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    backbone = "efficientnet_b1"
    if isinstance(checkpoint, dict):
        backbone = checkpoint.get("backbone", backbone)
        im_size = int(checkpoint.get("im_size", im_size))
    model = build_model(num_classes=len(label), backbone=backbone).to(device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


ensemble_models = [(load_single_model(name), float(weight)) for name, weight in MODEL_ENSEMBLE]


def _resize_rgb(X):
    if X.ndim == 2:
        X = np.repeat(X[:, :, None], 3, axis=2)
    if X.shape[2] == 4:
        X = X[:, :, :3]

    if cv2 is not None:
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        return cv2.resize(X, (im_size, im_size), interpolation=cv2.INTER_AREA)

    X = X[:, :, ::-1].copy()
    return np.asarray(Image.fromarray(X).resize((im_size, im_size)))


def _to_rgb_pil(X):
    if X.ndim == 2:
        X = np.repeat(X[:, :, None], 3, axis=2)
    if X.shape[2] == 4:
        X = X[:, :, :3]
    if cv2 is not None:
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    else:
        X = X[:, :, ::-1].copy()
    return Image.fromarray(X)


def _crop_pil(img, ratio, y_anchor):
    width, height = img.size
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_w) // 2)
    if y_anchor == "top":
        top = 0
    elif y_anchor == "bottom":
        top = max(0, height - crop_h)
    else:
        top = max(0, (height - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def _normalize_chw(X):
    X = (X - mean) / std
    return np.transpose(X, (2, 0, 1))


def _make_input_batch(X):
    X = _resize_rgb(X).astype(np.float32) / 255.0
    X = np.stack([_normalize_chw(X)], axis=0)
    return torch.from_numpy(X).to(device)


def _make_tta_input_batch(X):
    img = _to_rgb_pil(X)
    views = [
        img,
        _crop_pil(img, TTA_CROP_RATIO, "center"),
        _crop_pil(img, TTA_CROP_RATIO, "top"),
        _crop_pil(img, TTA_CROP_RATIO, "bottom"),
    ]
    arrays = []
    for view in views:
        view = np.asarray(view.resize((im_size, im_size))).astype(np.float32) / 255.0
        arrays.append(_normalize_chw(view))
    return torch.from_numpy(np.stack(arrays, axis=0)).to(device)


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str, 天气类别标签，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    X = _make_tta_input_batch(X) if USE_TTA else _make_input_batch(X)

    with torch.no_grad():
        prediction = None
        total_weight = 0.0
        for model, weight in ensemble_models:
            logits = model(X).mean(dim=0, keepdim=True)
            prediction = logits * weight if prediction is None else prediction + logits * weight
            total_weight += weight
        prediction = prediction / total_weight
    y_predict = label[int(torch.argmax(prediction, dim=1).item())]
    return y_predict
