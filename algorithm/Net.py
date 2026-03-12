import numpy as np
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_keypoints=14, input_channel=4):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 编码器
        self.backbone = nn.Sequential(
            # Stage 1: 128x64 -> 64x32
            nn.Conv2d(input_channel, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 2: 64x32 -> 32x16
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 3: 32x16 -> 16x8
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 4: 16x8 -> 8x4
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 解码器（上采样回高分辨率热图）
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, num_keypoints, kernel_size=1)  # 输出 14 张热图
        )

        # 高度预测分支
        self.height_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_keypoints)
        )

    def forward(self, x):
        """
        x: (B, 1, H, W) Range-Azimuth Map
        Returns:
            heatmaps: (B, K, H_out, W_out)  # 如 (B, 14, 32, 16)
            heights: (B, K)                  # 每个关键点的 z 坐标
        """
        feat = self.backbone(x)  # (B, 64, 8, 4)
        heatmaps = self.heatmap_head(feat)  # (B, K, 32, 16)
        heights = self.height_head(feat)  # (B, K)

        return feat,heatmaps, heights

model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)