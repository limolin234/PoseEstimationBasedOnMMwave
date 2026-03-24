import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
"""
Frame #	帧序号
# Obj	对象数量/ID
X, Y, Z	三维坐标
Doppler	多普勒速度
Intensity	信号强度
radar_avail_frames”：雷达估计关键点可用的帧数  98, 6481
“radar_est_kps”：雷达估计的关键点  6529,3,17
“gt_avail_frames”：用于三维人体关节的可用帧  66,6514
refined_gt_kps  精确点
"""
import pickle

#读cpl作为标签
data = pickle.load(open(r'D:\pytorch.pycharm\PoseEstimationBasedOnMMwave\algorithm\datasets\dataset_release\aligned_data\pose_labels\subject1_all_labels.cpl', 'rb'))
print(f"数据类型: {type(data)}")
if isinstance(data, dict):
    print("包含的字段 Keys:")
    for key in data.keys():
        print(f" - {key}")

    # 4. 查看具体某个字段的内容 (例如查看关键点数据)
    if 'refined_gt_kps' in data:
        # 假设它是 numpy 数组或列表
        kps = data['refined_gt_kps']
        print(f"形状/长度: {kps.shape if hasattr(kps, 'shape') else len(kps)}")
        # 打印前两个元素看看结构
        if hasattr(kps, '__getitem__'):
            print(kps[65:67])

#读csv作为原始数据
