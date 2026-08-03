import numpy as np
from typing import List, Union

def get_tool_position(joint_angles: np.ndarray, 
                      tool_length: float = 0.0) -> np.ndarray:
    """
    计算末端工具位姿（相对于基座标系）
    
    Args:
        joint_angles: 6个关节角度，单位弧度 [j1, j2, j3, j4, j5, j6]
        tool_length: 末端工具长度，单位mm (默认: 0.0)
    
    Returns:
        np.ndarray: [x, y, z, roll, pitch, yaw]，位置单位mm，角度单位度
    """
    PI = np.pi
    RADIAN = 180 / PI
    
    # DH参数
    a = np.array([0, 0, 285.03, -21.98, 0, 0])
    alpha = np.array([0, -PI/2, 0, PI/2, -PI/2, PI/2])
    theta = np.array([0, -PI * 172.22/180, -102.78/180 * PI, 0, 0, 0])
    d = np.array([123, 0, 0, 250.75, 0, 91])
    
    joint_angles = np.array(joint_angles)
    T_total = np.eye(4)
    
    # 计算正运动学
    for i in range(6):
        theta_i = joint_angles[i] + theta[i]
        calpha = np.cos(alpha[i])
        salpha = np.sin(alpha[i])
        ctheta = np.cos(theta_i)
        stheta = np.sin(theta_i)
        
        T_i = np.array([
            [ctheta, -stheta, 0, a[i]],
            [stheta * calpha, ctheta * calpha, -salpha, -salpha * d[i]],
            [stheta * salpha, ctheta * salpha, calpha, calpha * d[i]],
            [0, 0, 0, 1]
        ])
        T_total = T_total @ T_i
    
    # 添加工具长度（沿末端Z轴）
    tool_T = np.eye(4)
    tool_T[2, 3] = tool_length
    T_final = T_total @ tool_T
    
    # 提取位置
    result = np.zeros(6)
    result[0:3] = T_final[0:3, 3] / 100
    
    # 提取欧拉角 (roll, pitch, yaw)
    if T_final[2, 0] < -1 + 0.0001:
        result[4] = PI / 2 * RADIAN  # pitch
        result[5] = 0  # yaw
        result[3] = np.arctan2(T_final[1, 0], T_final[1, 1]) * RADIAN  # roll
    elif T_final[2, 0] > 1 - 0.0001:
        result[4] = -PI / 2 * RADIAN  # pitch
        result[5] = 0  # yaw
        result[3] = -np.arctan2(T_final[1, 0], T_final[1, 1]) * RADIAN  # roll
    else:
        _bt = np.arctan2(-T_final[2, 0], np.sqrt(T_final[0, 0]**2 + T_final[1, 0]**2))
        result[4] = _bt * RADIAN
        result[5] = np.arctan2(T_final[1, 0] / np.cos(_bt), T_final[0, 0] / np.cos(_bt)) * RADIAN
        result[3] = np.arctan2(T_final[2, 1] / np.cos(_bt), T_final[2, 2] / np.cos(_bt)) * RADIAN
    
    return result