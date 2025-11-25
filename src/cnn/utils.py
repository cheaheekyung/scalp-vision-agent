from __future__ import annotations

from typing import Dict, List

import torch


# 라벨 키 순서 고정 (항상 같은 순서로 [B, 6]을 만들기 위해)
VALUE_KEYS: List[str] = [f"value_{i}" for i in range(1, 7)]


def labels_dict_to_tensor(labels: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    ScalpDataset 배치에서 온 labels dict를 [B, 6] 텐서로 변환.

    Args:
        labels: {
            "value_1": Tensor[B],
            ...
            "value_6": Tensor[B]
        }

    Returns:
        targets: Tensor[B, 6] (각 원소는 0~3 정수)
    """
    tensors: List[torch.Tensor] = []

    for key in VALUE_KEYS:
        t = labels[key]          # 예상 shape: [B]
        # 혹시 [B]가 아닌 경우(예: [B, 1])를 대비해서 한 번 평탄화
        t = t.view(-1)
        tensors.append(t.long())  # CrossEntropyLoss는 long 타입 필요

    # [6, B] -> [B, 6]
    stacked = torch.stack(tensors, dim=1)
    return stacked
