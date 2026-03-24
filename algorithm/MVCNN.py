import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
import numpy as np


class MVCNNWithSparseConv(nn.Module):

    def __init__(self,
                 num_classes=40,
                 num_views=12,  # 渲染视角数
                 view_size=224,  # 每个视角的图像大小
                 backbone='resnet18',  # 2D CNN骨干网络
                 sparse_feat_dim=512,  # 稀疏卷积特征维度
                 grid_size=8):  # 3D空间划分网格数
        super().__init__()

        self.num_views = num_views
        self.view_size = view_size
        self.grid_size = grid_size
        self.grid_volume = grid_size ** 3
        if backbone == 'resnet18':
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.cnn_out_channels = 512  # ResNet18输出通道
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.view_fc = nn.Linear(self.cnn_out_channels, 256)
        self.sparse_conv_layers = nn.ModuleList([
            spconv.SparseConv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            spconv.SparseConv3d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            spconv.SparseConv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            spconv.SparseConv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        ])

        # ========== 阶段5：全局特征聚合 ==========
        # 稀疏全局池化
        self.global_pool = spconv.SparseGlobalPooling()

        # ========== 阶段6：分类头 ==========
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 视角池化权重（可学习）
        self.view_weights = nn.Parameter(torch.ones(num_views, 1, 1))

    def render_multiview(self, point_cloud):
        """
        将点云渲染成多视角图像
        注意：这是一个简化版本，实际需要用PyTorch3D等库

        Args:
            point_cloud: (B, N, 3) 点云坐标

        Returns:
            views: (B, num_views, 3, H, W) 多视角图像
        """
        # 这是一个占位函数
        # 实际实现需要：
        # 1. 定义相机位置（环绕物体）
        # 2. 渲染深度图或RGB图
        # 3. 返回多视角图像张量
        B, N, _ = point_cloud.shape
        return torch.randn(B, self.num_views, 3, self.view_size, self.view_size)

    def project_to_3d(self, view_features, point_cloud):
        """
        将多视角特征反投影到3D空间

        Args:
            view_features: (B, num_views, 256, H, W) 视角特征图
            point_cloud: (B, N, 3) 原始点云坐标

        Returns:
            sparse_feat: spconv.SparseConvTensor 稀疏3D特征
        """
        B, V, C, H, W = view_features.shape
        N = point_cloud.shape[1]
        device = point_cloud.device

        # 1. 将点云坐标量化到网格索引
        # 归一化到[-1, 1]
        normalized = point_cloud  # 假设已经在[-1,1]

        # 量化到[0, grid_size-1]
        grid_idx = ((normalized + 1) / 2 * (self.grid_size - 1e-5)).long()
        grid_idx = torch.clamp(grid_idx, 0, self.grid_size - 1)

        # 展平为1D索引
        flat_idx = (grid_idx[..., 0] * self.grid_size ** 2 +
                    grid_idx[..., 1] * self.grid_size +
                    grid_idx[..., 2])  # (B, N)

        # 2. 为每个点聚合多视角特征
        # 简单做法：取所有视角的平均（实际可用注意力机制）
        view_pooled = view_features.mean(dim=1)  # (B, C, H, W)

        # 3. 双线性插值采样视角特征
        # 将点投影到每个视角的图像平面，采样对应位置的特征
        # 简化：随机采样（实际需要相机参数）
        point_features = torch.randn(B, N, C, device=device)

        # 4. 构建稀疏特征
        # 去重：同一个网格可能有多个点
        unique_indices = []
        unique_features = []

        for b in range(B):
            # 获取当前batch的网格索引和特征
            batch_indices = flat_idx[b]  # (N,)
            batch_features = point_features[b]  # (N, C)

            # 去重（每个网格只保留一个点）
            unique_grids, inverse = torch.unique(batch_indices, return_inverse=True)

            # 对每个网格，聚合该网格内所有点的特征（取平均）
            grid_features = torch.zeros(len(unique_grids), C, device=device)
            grid_features.index_add_(0, inverse, batch_features)
            counts = torch.bincount(inverse).float().unsqueeze(1)
            grid_features = grid_features / counts

            # 记录坐标和特征
            for g_idx, grid_idx_val in enumerate(unique_grids):
                # 将1D索引转回3D
                z = grid_idx_val % self.grid_size
                y = (grid_idx_val // self.grid_size) % self.grid_size
                x = grid_idx_val // (self.grid_size ** 2)

                unique_indices.append([b, x, y, z])
                unique_features.append(grid_features[g_idx])

        # 转换为稀疏张量需要的格式
        indices = torch.tensor(unique_indices, dtype=torch.int32, device=device)
        features = torch.stack(unique_features, dim=0)  # (M, C)

        # 创建稀疏张量
        spatial_shape = [self.grid_size] * 3
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=B
        )

        return sparse_tensor

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3) 输入点云，已归一化到[-1,1]

        Returns:
            logits: (B, num_classes) 分类结果
        """
        B, N, _ = point_cloud.shape

        # ========== 阶段1：多视角渲染 ==========
        # views: (B, num_views, 3, H, W)
        views = self.render_multiview(point_cloud)

        # ========== 阶段2：2D CNN提取单视角特征 ==========
        # 将所有视角合并到batch维度处理
        views_flat = views.view(B * self.num_views, 3, self.view_size, self.view_size)

        # 通过CNN骨干
        view_feat_maps = self.cnn_backbone(views_flat)  # (B*V, C, H', W')

        # 全局平均池化得到视角特征向量
        view_feat_vectors = F.adaptive_avg_pool2d(view_feat_maps, 1).squeeze(-1).squeeze(-1)
        view_feat_vectors = view_feat_vectors.view(B, self.num_views, -1)  # (B, V, 512)

        # 特征降维
        view_features = self.view_fc(view_feat_vectors)  # (B, V, 256)

        # ========== 阶段3：视角池化（带可学习权重） ==========
        # 加权平均所有视角
        view_weights = F.softmax(self.view_weights, dim=0)
        pooled_view = (view_features * view_weights).sum(dim=1)  # (B, 256)

        # ========== 阶段4：反投影到3D空间 ==========
        # 将特征图reshape回空间维度
        view_feat_maps = view_feat_maps.view(B, self.num_views, 256,
                                             view_feat_maps.shape[-2],
                                             view_feat_maps.shape[-1])

        # 构建稀疏3D特征
        sparse_feat = self.project_to_3d(view_feat_maps, point_cloud)

        # ========== 阶段5：3D稀疏卷积处理 ==========
        x = sparse_feat
        for layer in self.sparse_conv_layers:
            if isinstance(layer, spconv.SparseConv3d):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                # BatchNorm对特征向量操作
                x = x.replace_feature(layer(x.features))
            elif isinstance(layer, nn.ReLU):
                x = x.replace_feature(layer(x.features))
            else:
                x = layer(x)

        # ========== 阶段6：全局特征聚合 ==========
        # 稀疏全局池化
        global_feat = self.global_pool(x)  # (B, 512)

        # ========== 阶段7：分类 ==========
        logits = self.classifier(global_feat)

        return logits


# ========== 简化版：适合FPGA部署的MVCNN ==========
class SimplifiedMVCNNFPGA(nn.Module):
    """
    进一步简化版本，去除复杂的反投影，直接用视角特征
    更适合FPGA部署
    """

    def __init__(self, num_classes=40, num_views=12):
        super().__init__()

        self.num_views = num_views

        # 简化的2D CNN（可以用深度可分离卷积）
        self.cnn_backbone = nn.Sequential(
            # 第1层
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第2层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第3层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 第4层
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 视角融合（可学习权重）
        self.view_weights = nn.Parameter(torch.ones(num_views, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def render_multiview(self, point_cloud):
        """简化渲染（占位）"""
        B, N, _ = point_cloud.shape
        return torch.randn(B, self.num_views, 3, 112, 112)

    def forward(self, point_cloud):
        B = point_cloud.shape[0]

        # 渲染多视角图像
        views = self.render_multiview(point_cloud)  # (B, V, 3, 112, 112)

        # 合并batch和视角维度
        views_flat = views.view(B * self.num_views, 3, 112, 112)

        # 2D CNN提取特征
        feat_maps = self.cnn_backbone(views_flat)  # (B*V, 256, 7, 7)

        # 全局平均池化
        feat_vectors = F.adaptive_avg_pool2d(feat_maps, 1).squeeze(-1).squeeze(-1)
        feat_vectors = feat_vectors.view(B, self.num_views, 256)  # (B, V, 256)

        # 视角融合（加权平均）
        view_weights = F.softmax(self.view_weights, dim=0)  # (V, 1)
        fused_feat = (feat_vectors * view_weights.unsqueeze(0)).sum(dim=1)  # (B, 256)

        # 分类
        logits = self.classifier(fused_feat)

        return logits


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 检查spconv是否安装
    try:
        import spconv

        print("spconv 已安装，可以使用完整版 MVCNNWithSparseConv")

        # 完整版模型
        model_full = MVCNNWithSparseConv(
            num_classes=40,
            num_views=12,
            grid_size=8
        )

        # 测试
        points = torch.randn(2, 1024, 3)
        with torch.no_grad():
            out = model_full(points)
        print(f"完整版输出形状: {out.shape}")

    except ImportError:
        print("spconv 未安装，使用简化版 SimplifiedMVCNNFPGA")

        # 简化版模型（无需spconv）
        model_simple = SimplifiedMVCNNFPGA(
            num_classes=40,
            num_views=12
        )

        # 测试
        points = torch.randn(2, 1024, 3)
        with torch.no_grad():
            out = model_simple(points)
        print(f"简化版输出形状: {out.shape}")
        print(f"简化版参数量: {sum(p.numel() for p in model_simple.parameters()):,}")