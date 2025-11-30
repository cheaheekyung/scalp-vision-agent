# src/cnn/inference.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms as T

from .cnn.models import MultiHeadResNet18  # 기존 학습에 쓰던 모델 그대로 import

from io import BytesIO
from PIL import Image

# =========================
# 설정
# =========================

# 프로젝트 루트 기준으로 맞춰서 수정해줘
# 예: scalp-vision-agent/models/multihead_resnet18_e4norm_v0.pth
MODEL_PATH = Path("notebooks/models/best_by_value6_acc.pt")

# ImageNet 정규화 (E4_norm 학습 때와 동일)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# 전역 객체 (앱 시작 시 1번만 로딩)
# =========================

_DEVICE: torch.device | None = None
_MODEL: torch.nn.Module | None = None
_TRANSFORM: T.Compose | None = None


def _get_device() -> torch.device:
    """CUDA가 있으면 cuda, 아니면 cpu."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _get_transform() -> T.Compose:
    """추론용 이미지 전처리 파이프라인 (Resize + CenterCrop + ToTensor + Normalize)."""
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return _TRANSFORM


def _build_model() -> torch.nn.Module:
    """
    MultiHeadResNet18 모델을 생성하고 weight 로드.
    - 학습 코드에서 사용한 설정과 동일해야 한다.
    """
    device = _get_device()

    # ✅ 학습 때 MultiHeadResNet18 생성할 때 사용한 인자를 그대로 맞춰줘야 함
    # 보통은 각 head가 4클래스(value_1~6 모두 0~3)라고 가정
    model = MultiHeadResNet18(
        num_classes_per_head=4
    )  # 이름/인자는 프로젝트에 맞게 조정

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    # 학습 코드에서 저장 방식에 따라 필요하면 key 정리 로직 추가
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def _get_model() -> torch.nn.Module:
    """전역 모델 객체를 반환 (없으면 로드)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


# =========================
# Public API
# =========================


def predict_condition_from_pil(image: Image.Image) -> Dict[str, int]:
    """
    PIL.Image 하나를 받아 value_1~value_6 등급(0~3)을 예측한다.

    반환 예시:
        {
            "value_1": 0,
            "value_2": 1,
            "value_3": 2,
            "value_4": 0,
            "value_5": 1,
            "value_6": 2,
        }
    """
    device = _get_device()
    model = _get_model()
    transform = _get_transform()

    # 전처리
    img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(img_tensor)  # 예상 shape: [1, 6, 4]

        # 만약 model이 dict나 리스트를 반환한다면, 여기에서 logits을 꺼내도록 수정하면 됨.
        if isinstance(logits, dict):
            # 예: {"logits": tensor}
            logits = logits["logits"]

        # [1, 6, 4] → [6, 4] → argmax → [6]
        if logits.dim() == 3 and logits.size(0) == 1:
            logits = logits[0]

        # 클래스 예측
        probs = torch.softmax(logits, dim=-1)  # [6, 4]
        preds = probs.argmax(dim=-1)  # [6]

    preds_cpu = preds.detach().cpu().tolist()  # 길이 6 리스트
    keys = [f"value_{i}" for i in range(1, 7)]

    return {k: int(v) for k, v in zip(keys, preds_cpu)}


def predict_condition_from_bytes(image_bytes: bytes) -> Dict[str, int]:
    """
    이미지 바이너리(bytes)를 받아 PIL.Image로 열고, predict_condition_from_pil을 호출한다.
    FastAPI UploadFile과 함께 쓰기 편한 형태.
    """

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return predict_condition_from_pil(image)
