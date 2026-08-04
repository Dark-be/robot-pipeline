"""
piper_ik 使用示例 (集成到你的 pipeline)

在你的代码中:
    import sys; sys.path.insert(0, "ik-geo/python")
    import piper_ik as pik

然后:
    kin = pik.build_kin()                       # 构建一次, 缓存复用
    Q, is_LS = pik.ik_spherical_physical(kin, R, p)   # 物理位姿 -> 所有逆解

示例: 验证 目标位姿 -> 逆解 -> 正解 闭环
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import piper_ik as pik


def rotation_from_rpy(roll, pitch, yaw):
    """ZYX 欧拉角 -> 旋转矩阵 (度->弧度)"""
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    pr, sp = np.radians(pitch), np.sin(np.radians(pitch))
    cp, cy, sy = np.cos(pr), np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    kin = pik.build_kin()

    # 1) 目标位姿: 位置(mm), 欧拉角(度)
    x, y, z = 200.0, 0.0, 350.0
    roll, pitch, yaw = 0.0, 90.0, 0.0
    R = rotation_from_rpy(roll, pitch, yaw)
    p = np.array([x, y, z])

    # 2) 逆解: 返回所有解 (最多 8 个)
    Q, is_LS = pik.ik_spherical_physical(kin, R, p)
    print(f"找到 {len(Q)} 组解:")
    for i, q in enumerate(Q):
        print(f"  解{i}: {np.round(np.degrees(q), 2)}  deg, 最小二乘={is_LS[i]}")

    # 3) 按当前关节角选最接近的解 (避免大范围跳跃)
    q_current = np.array([0.0, -1.57, 1.57, 0.0, -1.57, 0.0])
    best_idx = int(np.argmin([np.linalg.norm(q - q_current) for q in Q]))
    q_best = Q[best_idx]
    print(f"\n最优解(离当前关节角最近): {np.round(np.degrees(q_best), 2)} deg")

    # 4) 验证: 用正解检查该解能否复现目标位姿
    T = pik.fwdkin_dh(q_best)
    print(f"\n[闭环验证] 正解位置 = {np.round(T[:3, 3], 3)} mm (目标 {[x, y, z]})")
    print(f"           位置误差 = {np.linalg.norm(T[:3, 3] - p):.3e} mm")
    print(f"           旋转误差 = {np.linalg.norm(T[:3, :3] - R):.3e}")


if __name__ == "__main__":
    main()
