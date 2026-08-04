"""
Piper 机械臂 (球腕 6-DOF) 的 ik-geo 逆运动学求解器 (Python 版)

原理
----
ik-geo 用 6 个几何子问题 (subproblem) 来求解逆运动学。Piper 属于
"球腕 + 两平行轴" 族 (关节 2/3 平行, 关节 4/5/6 轴交于腕心), 可闭式求解:
  1. 用子问题 4 求解 q1
  2. 用子问题 3 求解 q3, 子问题 1 求解 q2
  3. 用子问题 4 求解 q5, 子问题 1 求解 q4 和 q6

本模块把已有的 Modified-DH 参数 (见 src/utils/kenimatics.py) 转换为
ik-geo 的 H/P (Product of Exponentials) 表示, 再调用 ik-geo 的子问题库
(linearSubproblemSltns) 求解。

坐标约定 (重要)
---------------
- 用户的 DH 为 Modified DH (Craig): T_i = Rot(x,α)·Trans(x,a)·Rot(z,θ)·Trans(z,d)
  -> 关节 i 绕 z_i 旋转
- ik-geo 的参考构型中末端旋转 = I, 因此需要把整机旋转 R_M^T
  (R_M 为 home 构型下末端姿态)
- 求解出的关节角 q 即用户 DH 中的 joint_angles (θ = offset + q)

依赖: numpy, 以及 ik-geo 的子问题库 (本仓库 ik-geo/python/subproblem_setups/libs)
"""
import sys
import os
import numpy as np

# 把 ik-geo 的 python 子问题库加入搜索路径
_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "subproblem_setups", "libs")
if _LIBS_DIR not in sys.path:
    sys.path.insert(0, _LIBS_DIR)

# 运行时动态加入搜索路径, Pylance 静态分析无法解析, 加 type: ignore
from sp1_lib import sp1_run  # type: ignore
from sp3_lib import sp3_run  # type: ignore
from sp4_lib import sp4_run  # type: ignore

# ============================================================
# 1. Piper Modified-DH 参数 (单位 mm, 与 src/utils/kenimatics.py 一致)
# ============================================================
A = np.array([0, 0, 285.03, -21.98, 0, 0])
ALPHA = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])
THETA_OFFSET = np.array([0, -np.pi*172.22/180, -102.78/180*np.pi, 0, 0, 0])
D = np.array([123, 0, 0, 250.75, 0, 91])

# ============================================================
# 2. 基础工具
# ============================================================
def rot(k, th):
    """Rodrigues 旋转矩阵"""
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def dh_transform(qi, i):
    """Modified DH 变换矩阵 (与 kenimatics.py 一致)"""
    ai, alfi, di = A[i], ALPHA[i], D[i]
    th = qi + THETA_OFFSET[i]
    cth, sth = np.cos(th), np.sin(th)
    ca, sa = np.cos(alfi), np.sin(alfi)
    return np.array([[cth, -sth, 0, ai],
                     [sth*ca, cth*ca, -sa, -sa*di],
                     [sth*sa, cth*sa, ca, ca*di],
                     [0, 0, 0, 1]])


def fwdkin_dh(q):
    """Modified-DH 正运动学, 返回 4x4 齐次变换"""
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(q[i], i)
    return T

# ============================================================
# 3. DH -> ik-geo H/P 转换
# ============================================================
def build_kin():
    """
    从 Modified-DH 参数构建 ik-geo 的 kin (H, P, joint_type)。
    返回 (H, P, R_M):
      H: 3x6 关节轴 (ik-geo 参考系)
      P: 3x7 关节位移 (P[:,i] 为关节 i 轴上的点, i=0..6, P[:,6] 为工具偏移)
      R_M: home 构型末端姿态 (用于物理位姿 <-> ik-geo 位姿转换)
    """
    n = 6
    # home 构型 (q=0, θ=offset) 下各 frame 的旋转与原点
    Rlist, plist = [np.eye(3)], [np.zeros(3)]
    T = np.eye(4)
    for i in range(n):
        T = T @ dh_transform(0, i)
        Rlist.append(T[:3, :3])
        plist.append(T[:3, 3])
    R_M = Rlist[n]

    # 关节 i 轴 = Modified-DH frame i 的 z 轴 (z_i)
    # 轴上的点 = frame i 的原点 p_{0,i}
    H = np.zeros((3, n))
    for i in range(n):
        H[:, i] = R_M.T @ Rlist[i+1][:, 2]

    # 球腕: 关节 4,5,6 轴交于腕心 = p_{0,4} (= p_{0,5})
    # 因此 P4=P5=P6=0 (轴4/5/6都过腕心), P7 = 工具偏移
    wrist = plist[4]
    P = np.zeros((3, n+1))
    P[:, 0] = R_M.T @ plist[1]                    # 关节1轴上的点
    P[:, 1] = R_M.T @ (plist[2] - plist[1])
    P[:, 2] = R_M.T @ (plist[3] - plist[2])
    P[:, 3] = R_M.T @ (wrist - plist[3])          # 关节4轴点 = 腕心
    P[:, 4] = 0.0                                 # 关节5轴点 = 腕心
    P[:, 5] = 0.0                                 # 关节6轴点 = 腕心
    P[:, 6] = R_M.T @ (plist[6] - wrist)          # 工具偏移 (frame6 坐标, 沿 z6=91)

    kin = {
        "H": H,
        "P": P,
        "joint_type": np.zeros(n),
        "R_M": R_M,
    }
    return kin


# ============================================================
# 4. ik-geo 风格正运动学 (用于验证)
# ============================================================
def fwdkin_ikgeo(kin, q):
    """ik-geo 递归正运动学, 返回 ik-geo 参考系下的 (R, p)"""
    H, P = kin["H"], kin["P"]
    p = P[:, 0].copy()
    R = np.eye(3)
    for i in range(H.shape[1]):
        R = R @ rot(H[:, i], q[i])
        p = p + R @ P[:, i+1]
    return R, p


def pose_physical_from_ikgeo(kin, q):
    """由 ik-geo 关节角得到物理位姿 (与 fwdkin_dh 一致)"""
    R_ik, p_ik = fwdkin_ikgeo(kin, q)
    R_M = kin["R_M"]
    return R_M @ R_ik, R_M @ p_ik

# ============================================================
# 5. 球腕逆运动学 (IK_spherical_2_parallel)
#    Piper 关节 2/3 平行 + 球腕, 属于 "球腕 + 两平行轴" 闭式族
# ============================================================
def _as_float_list(theta):
    """sp3/sp4 可能返回单个 float (最小二乘) 或数组, 统一为列表"""
    if np.isscalar(theta):
        return [float(theta)]
    return [float(t) for t in np.atleast_1d(np.asarray(theta, dtype=float))]


def ik_spherical_2_parallel(kin, R_06, p_0T):
    """
    球腕 + 两平行轴 机器人逆运动学 (闭式)。输入输出均在 ik-geo 参考系。
    Args:
        kin: build_kin() 的结果
        R_06: 3x3 末端姿态 (ik-geo 参考系)
        p_0T: 3x1 末端位置 (ik-geo 参考系)
    Returns:
        Q: Nx6 所有解 (每行一个 [q1..q6], 弧度)
        is_LS: Nx1 是否最小二乘解
    """
    H, P = kin["H"], kin["P"]

    # --- 子问题4: 求解 q1 ---
    # MATLAB: sp_4(H2, p, -H1, d)  =>  python sp4_run(p, k, h, d)
    t1, q1_ls = sp4_run(
        p_0T - R_06 @ P[:, 6] - P[:, 0], -H[:, 0], H[:, 1],
        H[:, 1] @ (P[:, 1] + P[:, 2] + P[:, 3]))

    Q = []
    is_LS = []
    for q1 in _as_float_list(t1):
        # --- 子问题3: 求解 q3 ---
        t3, q3_ls = sp3_run(
            -P[:, 3], P[:, 2], H[:, 2],
            np.linalg.norm(rot(-H[:, 0], q1) @ (-p_0T + R_06 @ P[:, 6] + P[:, 0])
                           + P[:, 1]))

        for q3 in _as_float_list(t3):
            # --- 子问题1: 求解 q2 ---
            q2, q2_ls = sp1_run(
                -P[:, 2] - rot(H[:, 2], q3) @ P[:, 3],
                rot(-H[:, 0], q1) @ (-p_0T + R_06 @ P[:, 6] + P[:, 0]) + P[:, 1],
                H[:, 1])

            R_36 = (rot(-H[:, 2], q3) @ rot(-H[:, 1], q2)
                    @ rot(-H[:, 0], q1) @ R_06)

            # --- 子问题4: 求解 q5 ---
            # MATLAB: sp_4(H4, H6, H5, d) => python sp4_run(p=H6, k=H5, h=H4, d)
            t5, q5_ls = sp4_run(
                H[:, 5], H[:, 4], H[:, 3], H[:, 3] @ R_36 @ H[:, 5])

            for q5 in _as_float_list(t5):
                # --- 子问题1: 求解 q4 ---
                q4, q4_ls = sp1_run(
                    rot(H[:, 4], q5) @ H[:, 5], R_36 @ H[:, 5], H[:, 3])
                # --- 子问题1: 求解 q6 ---
                q6, q6_ls = sp1_run(
                    rot(-H[:, 4], q5) @ H[:, 3], R_36.T @ H[:, 3], -H[:, 5])

                Q.append([q1, q2, q3, q4, q5, q6])
                is_LS.append(bool(q1_ls) or bool(q2_ls) or bool(q3_ls)
                             or bool(q4_ls) or bool(q5_ls) or bool(q6_ls))

    return np.array(Q), np.array(is_LS)


def ik_spherical_physical(kin, R_phys, p_phys):
    """
    物理位姿 -> 关节角。输入为物理坐标系 (基座标) 下的位姿。
    Args:
        R_phys: 3x3 物理末端姿态
        p_phys: 3x1 物理末端位置
    Returns:
        Q: Nx6 关节角解 (直接用于控制器, 弧度)
        is_LS: Nx1
    """
    R_M = kin["R_M"]
    # 物理位姿 -> ik-geo 参考系
    R_06 = R_M.T @ R_phys
    p_0T = R_M.T @ p_phys
    return ik_spherical_2_parallel(kin, R_06, p_0T)


# ============================================================
# 6. 自检
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    kin = build_kin()

    print("ik-geo 关节轴 H:")
    print(kin["H"])
    print("\nik-geo 关节位移 P:")
    print(kin["P"])

    # --- 正解一致性验证 ---
    rng = np.random.default_rng(0)
    max_eR, max_ep = 0.0, 0.0
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 6)
        T_dh = fwdkin_dh(q)
        R_phys, p_phys = pose_physical_from_ikgeo(kin, q)
        max_eR = max(max_eR, np.linalg.norm(T_dh[:3, :3] - R_phys))
        max_ep = max(max_ep, np.linalg.norm(T_dh[:3, 3] - p_phys))
    print("\n[自检1] fwdkin_dh vs ik-geo fwdkin: R误差=%.3e, p误差=%.3e" %
          (max_eR, max_ep))

    # --- 逆解验证 ---
    max_pos_err, max_rot_err = 0.0, 0.0
    n_ok = 0
    for trial in range(200):
        q_true = rng.uniform(-np.pi, np.pi, 6)
        T_dh = fwdkin_dh(q_true)
        R_phys, p_phys = T_dh[:3, :3], T_dh[:3, 3]

        Q, is_LS = ik_spherical_physical(kin, R_phys, p_phys)

        # 检查是否有解能复现目标位姿
        best = None
        for q in Q:
            Rr, pr = pose_physical_from_ikgeo(kin, q)
            err = np.linalg.norm(pr - p_phys) + np.linalg.norm(Rr - R_phys)
            if best is None or err < best:
                best = err
        if best is not None and best < 1e-6:
            n_ok += 1
        max_pos_err = max(max_pos_err, best)
    print("[自检2] 200 组随机位姿逆解: 可复现 %d/%d, 最大复现误差=%.3e" %
          (n_ok, 200, max_pos_err))
