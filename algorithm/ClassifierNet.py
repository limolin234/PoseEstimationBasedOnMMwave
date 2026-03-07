import torch.nn as nn
import torch
class PoseClassifier(nn.Module):
    """
    基于毫米波RAM图网络特征的姿态分类器
    支持两种特征输入：
    1. 3D关键点特征（14个关键点，每个x/y/z）
    2. 骨干网络卷积特征（64,8,4）
    """
    def __init__(self, num_keypoints=14, num_classes=4, feat_type="keypoint"):
        super().__init__()
        self.feat_type = feat_type  # "keypoint" 或 "backbone"
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        if feat_type == "keypoint":
            # 输入：(B, 14*3) 3D关键点展平
            self.fc_layers = nn.Sequential(
                nn.Linear(num_keypoints * 3, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),  # 低dropout适配小样本
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        elif feat_type == "backbone":
            # 输入：(B, 64*8*4) 骨干网络特征展平
            self.fc_layers = nn.Sequential(
                nn.Linear(64 * 8 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError("feat_type must be 'keypoint' or 'backbone'")

    def forward(self, feat):
        """
        feat: 特征输入
              - feat_type="keypoint": (B, num_keypoints*3)
              - feat_type="backbone": (B, 64,8,4)
        """
        if self.feat_type == "backbone":
            feat = torch.flatten(feat, start_dim=1)  # (B,64*8*4)
        logits = self.fc_layers(feat)
        return logits