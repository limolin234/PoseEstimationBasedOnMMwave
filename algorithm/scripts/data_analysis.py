"""
加特兰雷达AK/BK数据解析脚本（适配FPGA Python环境）
特点：轻量、无第三方复杂依赖（仅numpy，FPGA的Pynq/MicroPython均支持）
"""
import numpy as np
import serial  # FPGA端若用UART IP核，可替换为对应驱动（如pynq.lib.uart）

# ====================== 1. 配置参数（和雷达/软件一致，从加特兰手册获取） ======================
# 串口配置（FPGA端需替换为UART IP核的配置）
SERIAL_PORT = "COM3"  # FPGA端改为："/dev/ttyPS0"（Pynq）或UART设备号
BAUDRATE = 921600
TIMEOUT = 1

# 雷达协议配置（从加特兰软件采数结果/手册中确认）
FRAME_HEADER = b'\x55\xAA'  # AK帧头（加特兰常见，需核对）
AK_DATA_LEN = 20  # 每个AK点的数据长度（字节）：X(4)+Y(4)+Z(4)+SNR(2)+V(2)+预留(4)
FLOAT_BYTE_NUM = 4  # float32占4字节
INT16_BYTE_NUM = 2  # int16占2字节
BYTE_ORDER = 'little'  # 小端（加特兰雷达默认）

# 预处理配置（和PC端推理一致）
MIN_DISTANCE = 0.5  # 噪声过滤阈值（米）
MAX_RANGE = 3.0  # 最大检测距离（米）
INPUT_DIM = 100  # 模型输入固定维度


# ====================== 2. 核心解析函数（FPGA端可直接用） ======================
def check_frame_header(raw_data, header):
    """
    帧头校验：从原始字节流中找到有效帧的起始位置
    FPGA适配：简化版，只找第一个帧头（减少算力消耗）
    """
    header_len = len(header)
    for i in range(len(raw_data) - header_len + 1):
        if raw_data[i:i + header_len] == header:
            return i  # 返回帧头起始索引
    return -1  # 无有效帧


def parse_ak_data(raw_bytes):
    """
    解析AK点云数据：字节流 → X/Y/Z结构化数组
    输入：帧头后的AK数据字节流
    输出：(N,3)的X/Y/Z点云数组（N为有效点数）
    """
    # 1. 计算有效点数（总字节数 / 每个点的字节数）
    valid_point_num = len(raw_bytes) // AK_DATA_LEN
    if valid_point_num == 0:
        return np.array([])

    # 2. 逐点解析
    point_cloud = []
    for i in range(valid_point_num):
        # 截取单个点的字节段
        point_bytes = raw_bytes[i * AK_DATA_LEN: (i + 1) * AK_DATA_LEN]

        # 解析X/Y/Z（float32）
        x = np.frombuffer(point_bytes[0:FLOAT_BYTE_NUM], dtype=np.float32)[0]
        y = np.frombuffer(point_bytes[FLOAT_BYTE_NUM:2 * FLOAT_BYTE_NUM], dtype=np.float32)[0]
        z = np.frombuffer(point_bytes[2 * FLOAT_BYTE_NUM:3 * FLOAT_BYTE_NUM], dtype=np.float32)[0]

        # 解析SNR/V（可选，推理暂时不用）还需要确认 SNR 和 Velocity 的 分辨率 (Resolution) 和 偏移量 (Offset)。
        snr = int.from_bytes(point_bytes[12:14], byteorder=BYTE_ORDER)
        v = int.from_bytes(point_bytes[14:16], byteorder=BYTE_ORDER)

        point_cloud.append([x, y, z,snr,v])

    return np.array(point_cloud, dtype=np.float32)

# ====================== 3. 主函数（PC端验证/FPGA端部署） ======================
def radar_data_pipeline(is_fpga=False):
    """
    完整数据处理流水线：串口读取 → 解析 → 预处理
    is_fpga：是否为FPGA环境（True则适配FPGA的UART驱动）
    """
    # 1. 串口/硬件读取数据（FPGA端需替换驱动）
    if not is_fpga:
        # PC端：用pyserial读取串口
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=TIMEOUT)
        raw_data = ser.read(1024)  # 读取1024字节
        ser.close()
    else:
        # FPGA端：替换为UART IP核的读取逻辑（示例）
        # from pynq.lib import uart
        # uart_device = uart.UART('/dev/ttyPS0', BAUDRATE)
        # raw_data = uart_device.read(1024)
        raw_data = b''  # FPGA端需替换为实际读取逻辑

    # 2. 帧头校验
    header_idx = check_frame_header(raw_data, FRAME_HEADER)
    if header_idx == -1:
        print("无有效帧，返回空数据")
        return np.zeros((INPUT_DIM, 5), dtype=np.float32)

    # 3. 截取帧头后的数据段
    ak_bytes = raw_data[header_idx + len(FRAME_HEADER):]

    # 4. 解析点云
    point_cloud = parse_ak_data(ak_bytes)


    return point_cloud

#返回的是N*5的numpy数组
# ====================== 4. 测试（PC端验证解析逻辑） ======================
if __name__ == "__main__":
    # PC端运行：验证解析和预处理逻辑是否正确
    input_pts = radar_data_pipeline(is_fpga=False)
    print("解析并预处理后的点云形状：", input_pts.shape)
    print("前5个点数据：\n", input_pts[:5])