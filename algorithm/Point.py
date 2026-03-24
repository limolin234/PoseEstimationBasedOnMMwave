import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedPointNet(nn.Module):


    def __init__(self, num_classes=40, input_channels=3, grid_size=8):
        super().__init__()

        # 配置参数
        self.grid_size = grid_size
        self.grid_volume = grid_size ** 3
        self.spatial_range = (-1, 1)

        # ========== 第1阶段：独立点MLP ==========
        self.mlp1 = nn.Linear(input_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.mlp2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # ========== 第2阶段：全局特征提取==========
        self.fc1 = nn.Linear(self.grid_volume * 128, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)

        # ========== 第3阶段：分类头 ==========
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

    def grid_pooling(self, points, features):
        """
        局部网格池化：将空间划分成grid_size^3个网格，每个网格内取max

        Args:
            points: (B, N, 3) 点云坐标，已归一化到[-1, 1]
            features: (B, N, C) 每个点的特征

        Returns:
            grid_features: (B, grid_volume, C) 每个网格的聚合特征
        """
        B, N, C = features.shape
        device = points.device

        # 1. 将坐标从[-1,1]量化到[0, grid_size-1]
        # 公式: idx = floor((x + 1) / 2 * grid_size)
        normalized = (points + 1) / 2  # 从[-1,1] -> [0,1]
        grid_idx = (normalized * (self.grid_size - 1e-5)).long()  # (B, N, 3)
        grid_idx = torch.clamp(grid_idx, 0, self.grid_size - 1)

        # 2. 将3D索引展平为1D索引 (0 ~ grid_volume-1)
        flat_idx = (grid_idx[..., 0] * self.grid_size ** 2 +
                    grid_idx[..., 1] * self.grid_size +
                    grid_idx[..., 2])  # (B, N)

        # 3. 初始化网格特征为负无穷（用于max pooling）
        grid_features = torch.full((B, self.grid_volume, C),
                                   float('-inf'),
                                   device=device)

        # 4.  scatter max pooling
        # 对每个batch独立处理
        for b in range(B):
            # 获取当前batch的有效点（排除重复网格）
            indices = flat_idx[b]  # (N,)
            feat = features[b]  # (N, C)

            # 使用scatter_reduce实现max pooling
            # 对于每个网格，取该网格内所有点的特征的最大值
            grid_features[b] = torch.scatter_reduce(
                grid_features[b],  # 目标
                dim=0,  # 在网格维度上操作
                index=indices.unsqueeze(1).expand(-1, C),  # 索引广播到C维
                src=feat,
                reduce='amax',  # 最大值聚合
                include_self=False
            )

        # 5. 将负无穷替换为0（空网格用0填充）
        grid_features = torch.where(
            torch.isinf(grid_features),
            torch.zeros_like(grid_features),
            grid_features
        )

        return grid_features  # (B, grid_volume, C)

    def forward(self, points):
        """
        Args:
            points: (B, N, 3) 输入点云，已归一化到[-1, 1]

        Returns:
            logits: (B, num_classes) 分类结果
        """
        B, N, _ = points.shape

        # ========== 阶段1：独立点MLP ==========
        # 每个点独立通过MLP，可以完全并行
        x = points  # (B, N, 3)

        # MLP1: 3 -> 64
        x = self.mlp1(x)  # (B, N, 64)
        x = x.transpose(1, 2)  # (B, 64, N)  # BatchNorm1d需要通道在第二维
        x = self.bn1(x)
        x = x.transpose(1, 2)  # (B, N, 64)
        x = F.relu(x)

        # MLP2: 64 -> 128
        x = self.mlp2(x)  # (B, N, 128)
        x = x.transpose(1, 2)  # (B, 128, N)
        x = self.bn2(x)
        x = x.transpose(1, 2)  # (B, N, 128)
        point_features = F.relu(x)  # (B, N, 128)

        # ========== 阶段2：局部网格池化 ==========
        # 将空间划分成网格，每个网格内取max
        grid_features = self.grid_pooling(points, point_features)  # (B, 512, 128)

        # ========== 阶段3：全局特征提取 ==========
        # 将网格特征展平
        x = grid_features.reshape(B, -1)  # (B, 512*128)

        # FC1: 65536 -> 512
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC2: 512 -> 256
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        # ========== 阶段4：分类头 ==========
        x = self.fc3(x)  # (B, num_classes)

        return x


# ========== 辅助函数：数据预处理 ==========
def normalize_point_cloud(points):
    """
    将点云归一化到[-1, 1]区间
    Args:
        points: (N, 3) 原始点云
    Returns:
        normalized: (N, 3) 归一化后的点云
    """
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    max_dist = torch.max(torch.norm(points, dim=1))
    points = points / max_dist
    return points


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模型配置
    batch_size = 32
    num_points = 1024
    num_classes = 40

    # 创建模型
    model = SimplifiedPointNetFPGA(
        num_classes=num_classes,
        input_channels=3,
        grid_size=8  # 8x8x8 = 512个网格
    )

    # 模拟输入数据
    points = torch.randn(batch_size, num_points, 3)  # 随机点云

    # 前向传播
    logits = model(points)

    print(f"输入形状: {points.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试损失和反向传播
    labels = torch.randint(0, num_classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print("反向传播完成，梯度计算正常")