# src/cnn/models.py

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiHeadResNet18(nn.Module):
    """
    ResNet18 backbone + 멀티헤드 분류기.

    입력:
        x: [B, 3, 224, 224]

    출력:
        logits: [B, num_heads, num_classes_per_head]
    """

    def __init__(
        self,
        num_heads: int = 6,  # value_1 ~ value_6
        num_classes_per_head: int = 4,  # 등급 0~3
        use_pretrained: bool = True,  # ImageNet pretrained 사용할지 여부
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head
        self.use_pretrained = use_pretrained

        # 1) ResNet18 backbone 생성
        if self.use_pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = resnet18(weights=weights)

        # 2) 마지막 층 제거하고 feature extractor로 사용
        in_features = backbone.fc.in_features  # 보통 512
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # 3) 각 증상(value_1~value_6)을 위한 head num_heads개 생성
        self.heads = nn.ModuleList(
            [
                nn.Linear(in_features, self.num_classes_per_head)
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] (보통 H=W=224)

        Returns:
            logits: [B, num_heads, num_classes_per_head]
        """
        # ResNet18 backbone 통과 → [B, 512]
        features = self.backbone(x)

        # 각 head별로 logits 계산
        head_logits: Sequence[torch.Tensor] = [
            head(features) for head in self.heads
        ]  # 각 텐서: [B, num_classes_per_head]

        # [B, num_heads, num_classes_per_head]
        logits = torch.stack(head_logits, dim=1)
        return logits


if __name__ == "__main__":
    # 간단 shape 테스트용
    model = MultiHeadResNet18()
    x = torch.randn(8, 3, 224, 224)
    logits = model(x)
    print("logits shape:", logits.shape)
