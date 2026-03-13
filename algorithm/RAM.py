import numpy as np
import torch
from collections import deque
import torch.nn as nn
FRAME_QUEUE = deque(maxlen=4)
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


def ram_for_net(ram_map):
    """
    专门针对 pointcloud_to_ram 输出的 SNR 热图进行预处理
    输入: ram_map (128, 64) numpy array, 值为 SNR (dB)
    输出: tensor (1, 1, 128, 64), 归一化到 0~1 之间， ready for GPU
    """

    # 1. 处理全零情况 (没有检测到点)S
    if np.max(ram_map) < 1e-6:
        # 如果没有点，返回全零张量，避免除以零
        return torch.zeros((1, 1, 128, 64), dtype=torch.float32)

    # 2. 【核心】动态分位数截断 (Robust Normalization)
    # 理由：SNR 分布长尾，偶尔会有极大的噪点。
    # 我们忽略掉最高的 2% 的极端值，以第 98 百分位数为基准进行缩放。
    # 这样既保留了人体信号，又不会被单个噪点把整体压缩得太小。
    p98 = np.percentile(ram_map, 98)

    # 防止 p98 过小 (比如只有噪声时)
    if p98 < 1.0:
        p98 = np.max(ram_map) if np.max(ram_map) > 0 else 1.0

    # 截断：所有大于 p98 的值都压扁到 p98
    ram_clipped = np.clip(ram_map, 0, p98)

    # 归一化到 0 ~ 1
    ram_norm = ram_clipped / p98

    # 3. 转 Tensor & 浮点型
    tensor = torch.from_numpy(ram_norm).float()

    # 4. 增加维度 (Batch, Channel, H, W) -> (1, 1, 128, 64)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # 5. (可选) 进一步映射到 -1 ~ 1
    # 很多 CNN 结构在输入为 -1~1 时收敛更快。如果你的网络效果不好，可以解开下面这行
    # tensor = (tensor - 0.5) * 2.0

    return tensor

def get_stacked_ram_input(new_ram_map):
    """
    输入：新帧RAM图 (128,64)
    输出：堆叠后的4通道输入 (1,4,128,64)
    """
    # 单帧归一化
    single_frame = ram_for_net(new_ram_map)
    # 加入队列
    FRAME_QUEUE.append(single_frame)
    # 队列不足4帧时，用首帧填充（避免维度错误）
    while len(FRAME_QUEUE) < 4:
        FRAME_QUEUE.append(FRAME_QUEUE[0])
    # 拼接成多通道：(4,1,128,64) → (1,4,128,64)
    stacked_frames = torch.cat(list(FRAME_QUEUE), dim=1)
    return stacked_frames