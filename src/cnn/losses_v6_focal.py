# src/cnn/losses_v6_focal.py
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# 기본 CE (value_1~5에 사용)
_ce = nn.CrossEntropyLoss()


def focal_loss_multiclass(
    logits: torch.Tensor,   # [B, C]
    targets: torch.Tensor,  # [B]
    alpha: Optional[torch.Tensor] = None,  # [C] or None
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Multi-class Focal Loss (Lin et al., 2017).

    Args:
        logits: [B, C]
        targets: [B]
        alpha: 클래스별 weight 텐서 [C], 없으면 균등.
        gamma: focusing parameter (보통 1~3 사이)

    Returns:
        scalar loss tensor
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [B, C], got {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets shape [B], got {targets.shape}")

    # [B, C]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # p_t, log(p_t) 뽑기
    targets = targets.view(-1, 1)           # [B, 1]
    log_p_t = log_probs.gather(1, targets).squeeze(1)  # [B]
    p_t = probs.gather(1, targets).squeeze(1)          # [B]

    # alpha_t 설정
    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_t = alpha[targets.squeeze(1)]  # [B]
    else:
        alpha_t = 1.0

    # focal loss: - alpha_t * (1 - p_t)^gamma * log(p_t)
    loss = -alpha_t * (1 - p_t) ** gamma * log_p_t
    return loss.mean()


def multihead_ce_loss_v6_focal(
    logits: torch.Tensor,   # [B, H, C] = [B, 6, 4]
    targets: torch.Tensor,  # [B, H]     = [B, 6]
    gamma: float = 2.0,
    alpha_v6: Optional[torch.Tensor] = None,   # [4] or None
) -> torch.Tensor:
    """
    멀티헤드 출력에 대한 loss:
      - head 0~4 (value_1~value_5): 일반 CrossEntropyLoss
      - head 5 (value_6, 탈모): Focal Loss + 클래스 weight(alpha_v6)

    Args:
        logits: [B, num_heads, num_classes]
        targets: [B, num_heads]
        gamma: focal loss gamma (보통 2.0)
        alpha_v6: value_6용 클래스 weight [4].
                  None이면 기본값 [1.0, 1.5, 3.0, 4.0] 사용.

    Returns:
        total_loss: scalar tensor
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

    # 기본 alpha_v6 설정 (필요하면 여기 값만 조정하면서 실험하면 됨)
    if alpha_v6 is None:
        # 클래스 분포를 고려해 소수 클래스에 조금 더 weight 부여
        # (너무 과격하지 않게 적당한 수준으로 설정)
        alpha_v6 = torch.tensor(
            [1.0, 1.5, 3.0, 4.0],   # [class 0, 1, 2, 3]
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        alpha_v6 = alpha_v6.to(logits.device)

    per_head_losses: list[Tensor] = []

    for h in range(H):
        head_logits = logits[:, h, :]   # [B, C]
        head_targets = targets[:, h]    # [B]

        if h == H - 1:
            # 마지막 head = value_6 (탈모): focal loss 사용
            loss_h = focal_loss_multiclass(
                head_logits,
                head_targets,
                alpha=alpha_v6,
                gamma=gamma,
            )
        else:
            # 나머지 head: 기존 CrossEntropyLoss
            loss_h = _ce(head_logits, head_targets)

        per_head_losses.append(loss_h)

    losses_tensor = torch.stack(per_head_losses)  # [H]
    total_loss = losses_tensor.mean()
    return total_loss
