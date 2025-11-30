# src/cnn/run_training_efficientnet_vf.py
from __future__ import annotations

import os
import sys
import json
import time
import copy
from dataclasses import dataclass
from typing import Any, Tuple, Dict, List
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt

# --- 프로젝트 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MASTER_INDEX_CSV
from src.cnn.dataset import ScalpDataset
from src.cnn.models_efficientnet import MultiHeadEfficientNetB0
from src.cnn.losses import multihead_ce_loss
from src.cnn.train import get_device, train_one_epoch

# --- 유틸리티 클래스 및 함수 ---
HEAD_NAMES = ["value_1", "value_2", "value_3", "value_4", "value_5", "value_6"]
NUM_HEADS = len(HEAD_NAMES)
NUM_CLASSES = 4

@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best: float | None = None
    num_bad_epochs: int = 0

    def step(self, current: float) -> bool:
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

def extract_images_and_labels(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        images, labels_obj = batch["image"], batch["labels"]
    elif isinstance(batch, (list, tuple)):
        images, labels_obj = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if isinstance(labels_obj, torch.Tensor):
        labels = labels_obj
    elif isinstance(labels_obj, dict):
        tensors = [labels_obj[k] for k in sorted(labels_obj.keys())]
        labels = torch.stack(tensors, dim=1)
    else:
        raise TypeError(f"Unsupported labels type: {type(labels_obj)}")
    return images, labels

def compute_multihead_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def evaluate_one_epoch_with_heads(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, float, list[float], list[float]]:
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    head_losses = torch.zeros(NUM_HEADS, dtype=torch.float32)
    head_accs = torch.zeros(NUM_HEADS, dtype=torch.float32)

    for batch in dataloader:
        images, labels = extract_images_and_labels(batch)
        images, labels = images.to(device), labels.long().to(device)
        logits = model(images)
        
        loss = 0.0
        for h in range(NUM_HEADS):
            loss_h = F.cross_entropy(logits[:, h, :], labels[:, h])
            loss += loss_h
            head_losses[h] += loss_h.item()
            
            preds_h = logits[:, h, :].argmax(dim=-1)
            head_accs[h] += (preds_h == labels[:, h]).float().mean().item()

        total_loss += (loss / NUM_HEADS).item()
        total_acc += compute_multihead_accuracy(logits, labels)
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    head_losses /= n_batches
    head_accs /= n_batches
    return avg_loss, avg_acc, head_losses.tolist(), head_accs.tolist()


def main():
    """메인 학습 로직을 포함하는 함수"""
    
    # --- 주요 설정 ---
    EXPERIMENT_NAME = "E6_efficientnet_b0_vf"  # 실험 이름 변경
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LR = 1e-4
    USE_SUBSET = False

    # --- 결과 저장 경로 설정 ---
    results_root = PROJECT_ROOT / "results"
    results_dir = results_root / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved under: {results_dir.resolve()}")

    # --- 데이터셋 및 데이터로더 ---
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),  # 수직 뒤집기 추가
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = ScalpDataset(index_csv=MASTER_INDEX_CSV, split="train", transforms=train_transform)
    val_dataset = ScalpDataset(index_csv=MASTER_INDEX_CSV, split="val", transforms=val_transform)

    if USE_SUBSET:
        train_dataset = Subset(train_dataset, list(range(100)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train/Val dataset size: {len(train_dataset)} / {len(val_dataset)}")

    # --- 모델, 옵티마이저, 스케줄러 ---
    device = get_device()
    model = MultiHeadEfficientNetB0()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    # --- 메인 학습 루프 ---
    print(f"Starting training for experiment: {EXPERIMENT_NAME}")

    early_stopper = EarlyStopping(patience=3, min_delta=1e-4)
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_value6_acc": [], "val_head_losses": [], "val_head_accs": [],
    }
    best_val_loss = float("inf")
    best_value6_acc = 0.0
    best_state_by_loss: dict[str, Any] | None = None
    best_state_by_value6: dict[str, Any] | None = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== [{EXPERIMENT_NAME}] Epoch {epoch} / {NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, raw_head_losses, raw_head_accs = evaluate_one_epoch_with_heads(model, val_loader, device)

        val_head_losses = {name: float(v) for name, v in zip(HEAD_NAMES, raw_head_losses)}
        val_head_accs = {name: float(v) for name, v in zip(HEAD_NAMES, raw_head_accs)}
        value6_acc = float(val_head_accs.get("value_6", 0.0))

        print(f"[Summary] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, value_6_acc={value6_acc:.4f}")
        
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_value6_acc"].append(value6_acc)
        history["val_head_losses"].append(val_head_losses)
        history["val_head_accs"].append(val_head_accs)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = float(val_loss)
            best_state_by_loss = copy.deepcopy(model.state_dict())
            print(f"  ↳ Best (val_loss) model updated! val_loss={best_val_loss:.4f}")

        if value6_acc > best_value6_acc + 1e-4:
            best_value6_acc = value6_acc
            best_state_by_value6 = copy.deepcopy(model.state_dict())
            print(f"  ↳ Best (value_6) model updated! value_6_acc={best_value6_acc:.4f}")

        if early_stopper.step(val_loss):
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            break

    # --- 학습 종료 후 결과 저장 ---
    print(f"\n===== [{EXPERIMENT_NAME}] Training finished =====")

    # 1. History JSON 저장
    history_path = results_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✔ Saved history to {history_path.name}")

    # 2. 모델 가중치 저장
    torch.save(model.state_dict(), results_dir / "last_epoch.pt")
    print(f"✔ Saved last-epoch model to last_epoch.pt")
    if best_state_by_loss:
        torch.save(best_state_by_loss, results_dir / "best_by_val_loss.pt")
        print(f"✔ Saved best-by-loss model to best_by_val_loss.pt")
    if best_state_by_value6:
        torch.save(best_state_by_value6, results_dir / "best_by_value6_acc.pt")
        print(f"✔ Saved best-by-value6 model to best_by_value6_acc.pt")

    # 3. 성능 지표 CSV 저장
    metrics_df = pd.DataFrame({
        "epoch": history["epoch"], "train_loss": history["train_loss"], "train_acc": history["train_acc"],
        "val_loss": history["val_loss"], "val_acc": history["val_acc"], "val_value6_acc": history["val_value6_acc"],
    })
    metrics_df.to_csv(results_dir / "metrics_overall.csv", index=False)
    print(f"✔ Saved overall metrics to metrics_overall.csv")

    head_acc_rows = [{"epoch": ep, **accs} for ep, accs in zip(history["epoch"], history["val_head_accs"]) ]
    pd.DataFrame(head_acc_rows).set_index("epoch").to_csv(results_dir / "metrics_val_head_acc.csv")
    print(f"✔ Saved per-head val_acc metrics to metrics_val_head_acc.csv")

    # 4. 시각화 차트 저장
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("Warning: Malgun Gothic font not found. Skipping font setting.")


    epochs = history["epoch"]
    # Loss 곡선
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], ".-", label="train_loss")
    plt.plot(epochs, history["val_loss"], ".-", label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"[{EXPERIMENT_NAME}] Train vs Val Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(results_dir / "plot_loss.png"); plt.close()
    print(f"✔ Saved plot_loss.png")

    # Accuracy 곡선
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], ".-", label="train_acc")
    plt.plot(epochs, history["val_acc"], ".-", label="val_acc")
    plt.plot(epochs, history["val_value6_acc"], ".-", label="val_value6_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"[{EXPERIMENT_NAME}] Train/Val Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(results_dir / "plot_acc.png"); plt.close()
    print(f"✔ Saved plot_acc.png")

    # Head별 Val Accuracy 곡선
    plt.figure(figsize=(7, 5))
    for head in HEAD_NAMES:
        vals = [h.get(head, float("nan")) for h in history["val_head_accs"]]
        plt.plot(epochs, vals, ".-", label=head)
    plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title(f"[{EXPERIMENT_NAME}] Per-head Validation Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(results_dir / "plot_val_head_acc.png"); plt.close()
    print(f"✔ Saved plot_val_head_acc.png")

    print("\nAll tasks completed successfully!")


if __name__ == '__main__':
    main()
