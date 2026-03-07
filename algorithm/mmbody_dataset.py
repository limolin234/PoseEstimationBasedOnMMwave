import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from keypoints_utils import SMPL_19_TO_14, kpts3d_to_ram_coords, generate_gaussian_heatmap
# 假设你的 pointcloud_to_ram 在 init_model.py 或 utils.py 中
from justuse import pointcloud_to_ram  # ← 替换为你的实际导入路径


class MmBodyDataset(Dataset):
    def __init__(self, root_dir, split="train", heatmap_size=(32, 16)):
        self.root_dir = root_dir
        self.heatmap_size = heatmap_size

        # 获取所有点云文件（支持多级子目录）
        self.pc_files = []
        for subject in os.listdir(os.path.join(root_dir, "pointclouds")):
            subject_path = os.path.join(root_dir, "pointclouds", subject)
            if not os.path.isdir(subject_path):
                continue
            for action in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action)
                if os.path.isdir(action_path):
                    self.pc_files.extend(glob.glob(os.path.join(action_path, "*.npy")))

        self.pc_files = sorted(self.pc_files)
        self.kpt_files = [
            f.replace("pointclouds", "keypoints_3d") for f in self.pc_files
        ]

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        # 加载点云 (N, 5)
        pc = np.load(self.pc_files[idx]).astype(np.float32)
        # 加载关键点 (19, 3)
        kpts_19 = np.load(self.kpt_files[idx]).astype(np.float32)
        # 映射到 14 个关键点
        kpts_14 = kpts_19[SMPL_19_TO_14]  # (14, 3)

        # 生成 RAM (128, 64)
        ram = pointcloud_to_ram(pc, range_bins=128, angle_bins=64, max_range=5.0)

        # 生成热图标签
        ram_coords = kpts3d_to_ram_coords(kpts_14, max_range=5.0)  # (14, 2)
        heatmaps = np.stack([
            generate_gaussian_heatmap(coord, output_size=self.heatmap_size)
            for coord in ram_coords
        ], axis=0)  # (14, H, W)

        heights = kpts_14[:, 2]  # (14,)

        return {
            'ram': torch.from_numpy(ram).unsqueeze(0).float(),  # (1, 128, 64)
            'heatmaps': torch.from_numpy(heatmaps).float(),  # (14, H, W)
            'heights': torch.from_numpy(heights).float()  # (14,)
        }# mmbody_dataset.py
