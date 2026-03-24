import numpy as np


def preprocess(point_cloud,
               min_dist=0.5,  # 最小距离：过滤雷达近场杂波
               max_range=5.0,  # 最大距离：人体可能在较远处
               min_snr=4.0,  # 【核心】SNR 阈值：过滤背景噪声，保留人体(含四肢)
               target_points=200):  # 固定输出点数：人体结构复杂，建议 200+
    """
    人体姿态雷达点云预处理 (适配热图生成)
    :param point_cloud: 输入数组 (N, 5)，列顺序 [x, y, z, snr, v]
    :return: 清洗后的数组 (target_points, 5)，保留完整特征供热图使用
    """

    # 1. 空数据处理
    if point_cloud is None or len(point_cloud) == 0:
        return np.zeros((target_points, 5), dtype=np.float32)

    # 2. 提取各列 (确保输入是 5 列)
    # 如果输入只有 3 列，这里会报错，请确保上游数据是 5 列
    xyz = point_cloud[:, :3]
    snr = point_cloud[:, 3]
    # vel = point_cloud[:, 4] # 速度保留在数据中，不参与过滤

    # 3. 【修正】只基于 x,y,z 计算物理距离
    distances = np.linalg.norm(xyz, axis=1)

    # 4. 构建综合过滤掩码
    # 条件 A: 距离过滤 (去掉太近和太远)
    mask_dist = (distances > min_dist) & (distances < max_range)

    # 条件 B: SNR 过滤 (去掉弱噪声，保留人体信号)
    mask_snr = snr > min_snr

    # 合并条件
    final_mask = mask_dist & mask_snr
    filtered_pts = point_cloud[final_mask]

    # 5. 无有效点处理
    if len(filtered_pts) == 0:
        return np.zeros((target_points, 5), dtype=np.float32)

    # 6. 固定点数 (采样或填充)
    num_pts = len(filtered_pts)

    if num_pts > target_points:
        # 点太多：随机采样 (保持分布代表性)
        indices = np.random.choice(num_pts, target_points, replace=False)
        final_pts = filtered_pts[indices]
    else:
        # 点太少：零填充 (Padding)
        final_pts = np.zeros((target_points, 5), dtype=np.float32)
        final_pts[:num_pts] = filtered_pts

    # 7. 【重要】不进行坐标归一化，直接返回物理坐标
    # 热图生成函数 (pointcloud_to_ram) 需要根据真实坐标 (米) 来映射像素
    return final_pts
#返回200*5的numpy数组