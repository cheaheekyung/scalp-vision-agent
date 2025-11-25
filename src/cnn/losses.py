from __future__ import annotations

import torch
from torch import nn


_ce = nn.CrossEntropyLoss()


def multihead_ce_loss(
    logits: torch.Tensor,   # [B, H, C] = [B, 6, 4]
    targets: torch.Tensor,  # [B, H]     = [B, 6]
) -> torch.Tensor:
    """
    멀티헤드 출력에 대해 증상별 CrossEntropyLoss를 계산하고 평균을 반환.

    Args:
        logits: [B, num_heads, num_classes_per_head]
        targets: [B, num_heads]  (각 원소는 0~C-1)

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

    per_head_losses = []

    for h in range(H):
        head_logits = logits[:, h, :]   # [B, C]
        head_targets = targets[:, h]    # [B]
        loss_h = _ce(head_logits, head_targets)
        per_head_losses.append(loss_h)

    # [H] -> 평균
    losses_tensor = torch.stack(per_head_losses)  # [H]
    total_loss = losses_tensor.mean()
    return total_loss
