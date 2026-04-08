from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms as T

from .cnn.models_efficientnet import MultiHeadEfficientNetB0


# 현재 서비스가 사용하는 최종 체크포인트
MODEL_PATH = Path("results/E8_efficientnet_b0_v6_focal/best_by_value6_acc.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 앱 프로세스 동안 재사용할 전역 캐시
_DEVICE: torch.device | None = None
_MODEL: torch.nn.Module | None = None
_TRANSFORM: T.Compose | None = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _get_transform() -> T.Compose:
    global _TRANSFORM
    if _TRANSFORM is None:
        # 학습 시 사용한 입력 규격에 맞춰 추론용 전처리 고정
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
    device = _get_device()

    # E8 실험과 동일한 EfficientNet 멀티헤드 구조를 사용
    model = MultiHeadEfficientNetB0(
        num_heads=6,
        num_classes_per_head=4,
        use_pretrained=False,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # 저장된 학습 가중치를 불러와 바로 추론 모드로 전환
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        # 첫 요청 시 한 번만 모델을 만들고 이후 요청에서는 재사용
        _MODEL = _build_model()
    return _MODEL


def predict_condition_from_pil(image: Image.Image) -> Dict[str, int]:
    device = _get_device()
    model = _get_model()
    transform = _get_transform()

    # 배치 차원을 추가해 모델 입력 형태 [1, 3, 224, 224]로 맞춤
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)

        if isinstance(logits, dict):
            logits = logits["logits"]

        if logits.dim() == 3 and logits.size(0) == 1:
            logits = logits[0]

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

    # 6개 head의 예측 클래스를 API 스키마 키 형식으로 변환
    preds_cpu = preds.detach().cpu().tolist()
    keys = [f"value_{i}" for i in range(1, 7)]
    return {k: int(v) for k, v in zip(keys, preds_cpu)}


def predict_condition_from_bytes(image_bytes: bytes) -> Dict[str, int]:
    # FastAPI UploadFile bytes를 바로 추론 함수에 연결하기 위한 진입점
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return predict_condition_from_pil(image)
