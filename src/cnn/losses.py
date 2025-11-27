from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

_ce = nn.CrossEntropyLoss()


def multihead_ce_loss(
    logits: torch.Tensor,   # [B, H, C] = [B, 6, 4]
    targets: torch.Tensor,  # [B, H]     = [B, 6]
    # label_smoothing: float = 0.0,
    # head_weights: Tensor | None = None,   # [H] = [6] or None
) -> torch.Tensor:
    """
    멀티헤드 출력에 대해 증상별 CrossEntropyLoss를 계산하고 평균을 반환.

    Args:
        logits: [B, num_heads, num_classes_per_head]
        targets: [B, num_heads]  (각 원소는 0~C-1)
        label_smoothing: CrossEntropyLoss에 적용할 label smoothing 계수
        head_weights: head별 가중치 텐서 [H].
                      예: 탈모(value_6)만 1.5배 주고 싶으면 [1,1,1,1,1,1.5]

    Returns:
        total_loss: scalar 텐서
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, H, C], got {logits.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets shape [B, H], got {targets.shape}")

    B, H, C = logits.shape
    if targets.shape != (B, H):
        raise ValueError(
            f"Shape mismatch: logits {logits.shape}, targets {targets.shape}"
        )
    # if head_weights is not None:
    #     if head_weights.shape != (H,):
    #         raise ValueError(
    #             f"Expected head_weights shape [{H}], got {head_weights.shape}"
    #         )
    #     # 디바이스 맞춰주기
    #     head_weights = head_weights.to(logits.device)

    per_head_losses: list[Tensor] = []

    for h in range(H):
        head_logits = logits[:, h, :]   # [B, C]
        head_targets = targets[:, h]    # [B]
        loss_h = _ce(head_logits, head_targets)
        # loss_h =  F.cross_entropy(
        #     head_logits,
        #     head_targets,
        #     label_smoothing=label_smoothing,
        # )  # scalar
        per_head_losses.append(loss_h)

    # [H] -> 평균
    losses_tensor = torch.stack(per_head_losses)  # [H]

    # head별 가중치 적용 (예: 탈모 head만 1.5배)
    # if head_weights is not None:
    #     losses_tensor = losses_tensor * head_weights

    # 평균    
    total_loss = losses_tensor.mean()
    return total_loss
