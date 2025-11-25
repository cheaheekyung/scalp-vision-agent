from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.cnn.models import MultiHeadResNet18
from src.cnn.utils import labels_dict_to_tensor
from src.cnn.losses import multihead_ce_loss


def get_device() -> torch.device:
    """GPU가 있으면 cuda, 없으면 cpu 사용."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    한 epoch 동안 학습을 수행하는 함수.

    Returns:
        epoch_loss: 전체 데이터셋 기준 평균 loss
        epoch_acc:  모든 증상(head)을 포함한 전체 정확도 (0~1)
    """
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in dataloader:
        images = batch["image"].to(device)  # [B, 3, 224, 224]
        targets = labels_dict_to_tensor(batch["labels"]).to(device)  # [B, 6]

        optimizer.zero_grad()

        logits = model(images)  # [B, 6, 4]
        loss = multihead_ce_loss(logits, targets)

        loss.backward()
        optimizer.step()

        # --- 통계 계산용 ---
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # 예측 등급: [B, 6, 4] → [B, 6]
        preds = logits.argmax(dim=2)
        running_correct += (preds == targets).sum().item()
        running_total += targets.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / running_total  # 전체 (B*6) 중 맞은 개수 비율

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    검증용 평가 함수 (gradient 계산 X).

    Returns:
        epoch_loss: 평균 loss
        epoch_acc:  전체 정확도
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = labels_dict_to_tensor(batch["labels"]).to(device)

        logits = model(images)
        loss = multihead_ce_loss(logits, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=2)
        running_correct += (preds == targets).sum().item()
        running_total += targets.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc
