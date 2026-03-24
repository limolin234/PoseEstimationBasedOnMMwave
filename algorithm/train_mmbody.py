import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# 配置
DATA_ROOT = "/path/to/mmBody"  # ← 修改为你的 mmBody 路径
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
HEIGHT_LOSS_WEIGHT = 0.1  # 根据验证调整

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集 & 加载器
train_dataset = MmBodyDataset(DATA_ROOT, split="train", heatmap_size=(32, 16))
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 模型 & 损失 & 优化器
model = Net(num_keypoints=14).to(device)
heatmap_criterion = nn.MSELoss()
height_criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        ram = batch['ram'].to(device)
        heatmaps_gt = batch['heatmaps'].to(device)
        heights_gt = batch['heights'].to(device)

        # 前向传播
        heatmaps_pred, heights_pred = model(ram)

        # 计算损失
        loss_hm = heatmap_criterion(heatmaps_pred, heatmaps_gt)
        loss_ht = height_criterion(heights_pred, heights_gt)
        loss = loss_hm + HEIGHT_LOSS_WEIGHT * loss_ht

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "mmwave_pose_mmbody.pth")
print("Model saved to mmwave_pose_mmbody.pth")