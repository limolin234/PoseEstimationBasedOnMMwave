import numpy as np
import torch
import torch.nn as nn

def pointcloud_to_ram(points, range_bins=128, angle_bins=64, max_range=5.0):
    """
    将毫米波点云转换为 Range-Azimuth Map (RAM)
    输入: points (N, 5) -> [x, y, z, v, snr]
    输出: ram_map (range_bins, angle_bins) -> 每个 cell 存最大 SNR 或点数
    """
    x, y = points[:, 0], points[:, 1]

    # 计算距离和方位角
    ranges = np.sqrt(x ** 2 + y ** 2)
    angles = np.arctan2(y, x)  # [-π, π]

    # 过滤有效范围
    mask = (ranges <= max_range) & (ranges > 0.1)
    ranges = ranges[mask]
    angles = angles[mask]
    snr = points[mask, 4]

    # 离散化到网格
    range_idx = (ranges / max_range * range_bins).astype(int)
    angle_idx = ((angles + np.pi) / (2 * np.pi) * angle_bins).astype(int)

    # 限制索引范围
    range_idx = np.clip(range_idx, 0, range_bins - 1)
    angle_idx = np.clip(angle_idx, 0, angle_bins - 1)

    # 构建 RAM 图（这里用最大 SNR）
    ram_map = np.zeros((range_bins, angle_bins), dtype=np.float32)
    for r, a, s in zip(range_idx, angle_idx, snr):
        if s > ram_map[r, a]:
            ram_map[r, a] = s

    return ram_map  # (128, 64)


def heatmap_to_3d(heatmaps, heights, max_range=5.0, range_bins=128, angle_bins=64):
    """
    将热图峰值转换为 3D 关键点
    heatmaps: (K, H, W)  # 如 (14, 32, 16)
    heights: (K,)         # 预测的 z 坐标
    """
    keypoints_3d = []
    H, W = heatmaps.shape[1], heatmaps.shape[2]

    for k in range(heatmaps.shape[0]):
        hm = heatmaps[k].cpu().numpy()
        # 找峰值（可用 soft-argmax 提升精度）
        idx = np.argmax(hm)
        r_idx, a_idx = np.unravel_index(idx, (H, W))

        # 映射回原始 RAM 坐标系
        r_norm = (r_idx + 0.5) / H  # 归一化距离 [0, 1]
        a_norm = (a_idx + 0.5) / W  # 归一化角度 [0, 1]

        # 转换为物理量
        r = r_norm * max_range  # 米
        theta = a_norm * 2 * np.pi - np.pi  # [-π, π]

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = heights[k].item()

        keypoints_3d.append([x, y, z])

    return np.array(keypoints_3d)  # (K, 3)
class Net(nn.Module):
    def __init__(self, num_keypoints=14, input_size=(128, 64)):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 编码器（轻量，适合 FPGA）
        self.backbone = nn.Sequential(
            # Stage 1: 128x64 -> 64x32
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
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

        # 高度预测分支（可选，轻量）
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
