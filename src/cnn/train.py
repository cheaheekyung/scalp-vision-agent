from __future__ import annotations

from typing import Tuple, Dict
from pathlib import Path

import time

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

History = Dict[str, list[float]]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device | None = None,
    best_model_path: str | Path = "models/multihead_resnet18_best.pth",
    last_model_path: str | Path | None = "models/multihead_resnet18_last.pth",
) -> History:
    """
    여러 epoch 학습 + 검증을 수행하고,
    val_loss 기준으로 베스트 모델 가중치를 저장.

    Args:
        model: 학습할 모델
        train_loader: 학습용 DataLoader
        val_loader: 검증용 DataLoader
        optimizer: 옵티마이저
        num_epochs: 학습 epoch 수
        device: 사용할 디바이스 (None이면 자동 선택)
        best_model_path: val_loss가 가장 낮을 때마다 저장할 경로
        last_model_path: 마지막 epoch 모델을 저장할 경로 (원치 않으면 None)

    Returns:
        history: 각 epoch별 loss/acc 기록 dict
    """
    if device is None:
        device = get_device()

    model.to(device)

    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    if last_model_path is not None:
        last_model_path = Path(last_model_path)
        last_model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    history: History = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [], 
    }

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter() 

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)

        # 베스트 모델 저장 (val_loss 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ↳ Best model updated! val_loss={val_loss:.4f}")
        
        m, s = divmod(epoch_time, 60)
        time_str = f"{int(m):02d}:{int(s):02d}"

        print(
            f"[{epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            f"time={time_str} ({epoch_time:.1f}s)"
        )

    # 마지막 epoch 모델도 별도로 저장하고 싶다면
    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    return history