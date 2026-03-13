import torch
import torch.nn as nn
from data_analysis import radar_data_pipeline
from preprocess import preprocess
from Net import Net
from RAM import heatmap_to_3d,pointcloud_to_ram,get_stacked_ram_input
from ClassifierNet import PoseClassifier
"模型的推理"
# ====================== 1. 配置参数 ======================
# 串口配置
SERIAL_PORT = "COM3"  # FPGA端改为："/dev/ttyPS0"（Pynq）或UART设备号
BAUDRATE = 921600
TIMEOUT = 1

# 雷达协议配置
FRAME_HEADER = b'\x55\xAA'  # AK帧头（加特兰常见，需核对）
AK_DATA_LEN = 20  # 每个AK点的数据长度（字节）：X(4)+Y(4)+Z(4)+SNR(2)+V(2)+预留(4)
FLOAT_BYTE_NUM = 4  # float32占4字节
INT16_BYTE_NUM = 2  # int16占2字节
BYTE_ORDER = 'little'  # 小端（加特兰雷达默认）

# 预处理配置
MIN_DISTANCE = 0.5  # 噪声过滤阈值（米）
MAX_RANGE = 3.0  # 最大检测距离（米）
INPUT_DIM = 100  # 模型输入固定维度

#网络模型参数
range_bins=128
angle_bins=64
numpoints = 14   #关键节点数量
input_channels = 4  #输入帧数
numclasses = 5   #姿态类别
model = Net(num_keypoints=numpoints,input_channel=input_channels)
# model.load_state_dict(torch.load("Net_weights.pth", map_location="cpu"))
model.eval()
if __name__ == "__main__":
    frame_counter = 0
    while True:
        frame_counter += 1
        # ====================== 2. 数据解析 ======================
        in_put=radar_data_pipeline(is_fpga=False)
        if in_put.ndim != 2 or in_put.shape[1] != 5:
            print(f"雷达数据维度不符合预期跳过此帧...")
            continue
        # ====================== 3. 点云预处理 ======================
        RAM_input = preprocess(in_put,min_dist=MIN_DISTANCE, max_range=MAX_RANGE)#返回200*5的numpy数组
        # ====================== 4. 转热图 ======================
        model_input = pointcloud_to_ram(RAM_input)#(128, 64)的数组
        model_input = get_stacked_ram_input(model_input)  # tensor (1, 4, 128, 64)
        # ====================== 5. 特征提取 ======================
        with torch.no_grad():
            feat,heatmaps, heights = model(model_input)
        # 热图→3D关键点
        keypoints_3d = heatmap_to_3d(heatmaps[0], heights[0])
        print(f"   3D 骨架点形状: {keypoints_3d}")
        # ====================== 6. 姿态识别 ======================
        # 方式1：基于3D关键点特征分类
        # classifier_keypoint = PoseClassifier(feat_type="keypoint", num_classes=4)
        # classifier_keypoint.eval()
        # keypoint_feat = torch.from_numpy(keypoints_3d).flatten().unsqueeze(0).float()  # (1,42)
        # logits_keypoint = classifier_keypoint(keypoint_feat)  # (1,4)
        # out_put_1 = torch.argmax(logits_keypoint, dim=1).item()
        # # 方式2：基于骨干网络卷积特征分类
        # classifier_backbone = PoseClassifier(feat_type="backbone", num_classes=4)
        # classifier_backbone.eval()
        # logits_backbone = classifier_backbone(feat)  # (1,4)
        # out_put_2 = torch.argmax(logits_backbone, dim=1).item()





