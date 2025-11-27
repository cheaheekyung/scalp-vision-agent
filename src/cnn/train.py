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


# 탈모 head(value_6)에만 가중치 1.5배
# HEAD_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.5])


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
        loss = multihead_ce_loss(logits=logits, targets=targets)
        # 🔹 여기서 outputs → logits 로 수정
        # loss = multihead_ce_loss(
        #     logits=logits,
        #     targets=targets,
        #     label_smoothing=0.1,            # E1
        #     head_weights=HEAD_WEIGHTS,      # E2
        # )

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
        # 🔹 train과 동일하게 label_smoothing + head_weights 사용
        # loss = multihead_ce_loss(
        #     logits=logits,
        #     targets=targets,
        #     label_smoothing=0.1,
        #     head_weights=HEAD_WEIGHTS,
        # )

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
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"time={time_str} ({epoch_time:.1f}s)"
        )

    if last_model_path is not None:
        torch.save(model.state_dict(), last_model_path)

    return history
