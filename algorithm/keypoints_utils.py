# keypoints_utils.py
import numpy as np
import cv2

# SMPL 19 关节 → 自定义 14 关键点映射
SMPL_19_TO_14 = [
    0,  # nose
    16,  # left_shoulder
    17,  # right_shoulder
    18,  # left_elbow
    19,  # right_elbow
    20,  # left_wrist
    21,  # right_wrist
    1,  # left_hip
    2,  # right_hip
    4,  # left_knee
    5,  # right_knee
    7,  # left_ankle
    8,  # right_ankle
    15  # head_top
]


def kpts3d_to_ram_coords(kpts3d, max_range=5.0):
    """
    将 3D 关键点 (x, y, z) 转换为 RAM 归一化坐标 (a_norm, r_norm)
    kpts3d: (K, 3)
    Returns: (K, 2) -> [[a_norm, r_norm], ...]  # 注意顺序：角度在前，距离在后
    """
    x, y = kpts3d[:, 0], kpts3d[:, 1]
    ranges = np.sqrt(x ** 2 + y ** 2)
    angles = np.arctan2(y, x)  # [-π, π]

    r_norm = np.clip(ranges / max_range, 0, 1)
    a_norm = (angles + np.pi) / (2 * np.pi)  # [0, 1]

    return np.stack([a_norm, r_norm], axis=1)


def generate_gaussian_heatmap(coord, output_size=(32, 16), sigma=1.0):
    """
    coord: (a_norm, r_norm) in [0, 1]
    output_size: (H, W) = (range_bins_out, angle_bins_out)
    """
    H, W = output_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    # 转换为像素坐标 (注意：RAM 中 range 是行，angle 是列)
    px = int(coord[0] * W)  # angle → width (列)
    py = int(coord[1] * H)  # range → height (行)

    if 0 <= px < W and 0 <= py < H:
        # 创建高斯核
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        gaussian = cv2.getGaussianKernel(size, sigma)
        gaussian = gaussian @ gaussian.T

        # 放置到热图
        ul = (px - size // 2, py - size // 2)
        br = (ul[0] + size, ul[1] + size)

        g_ul = (max(0, -ul[0]), max(0, -ul[1]))
        g_br = (min(size, W - ul[0]), min(size, H - ul[1]))
        h_ul = (max(ul[0], 0), max(ul[1], 0))
        h_br = (min(br[0], W), min(br[1], H))

        heatmap[h_ul[1]:h_br[1], h_ul[0]:h_br[0]] = \
            np.maximum(heatmap[h_ul[1]:h_br[1], h_ul[0]:h_br[0]],
                       gaussian[g_ul[1]:g_br[1], g_ul[0]:g_br[0]])

    return heatmap