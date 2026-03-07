import numpy as np
def preprocess(point_cloud, min_dist=0.5, max_range=3.0, input_dim=100):
    """
    轻量化预处理函数（FPGA端直接用）
    :param point_cloud: 解析后的原始点云 (N,3)
    :return: 预处理后的模型输入 (input_dim,3)
    """
    if len(point_cloud) == 0:  # 无有效点，返回全0
        return np.zeros((input_dim, 3), dtype=np.float32)

    # 步骤1：过滤噪声
    distance = np.linalg.norm(point_cloud, axis=1)
    valid_pts = point_cloud[distance > min_dist]

    # 步骤2：归一化
    normalized_pts = valid_pts / max_range

    # 步骤3：固定维度
    input_pts = np.zeros((input_dim, 3), dtype=np.float32)
    fill_num = min(len(normalized_pts), input_dim)
    input_pts[:fill_num] = normalized_pts[:fill_num]

    return input_pts

