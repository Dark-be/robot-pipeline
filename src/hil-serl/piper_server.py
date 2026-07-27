"""
HIL-SERL Robot Server 模版
=========================
替换成你自己的机械臂硬件协议即可接入 HIL-SERL 训练。

Gym Env 通过 HTTP POST 与此 server 通信。所有路由均为 POST。
必须实现的端点已标注 [REQUIRED]，可选端点标注 [OPTIONAL]。
"""

from flask import Flask, request, jsonify
import numpy as np
import time


# ============================================================================
# Robot Server 主类
# ============================================================================

class RobotServer:
    """
    机械臂的硬件无关接口，负责：
    - 发送末端位姿指令 (/pose)
    - 回读完整状态 (/getstate)
    - 清除错误 (/clearerr)
    - 关节复位 (/jointreset)   [REQUIRED 仅当使用 joint_reset_period]
    - 切换控制参数 (/update_param) [REQUIRED]
    """

    def __init__(self):
        # ---------- 内部状态缓存 ----------
        self.pose = np.zeros(7)        # [x, y, z, qx, qy, qz, qw]
        self.vel = np.zeros(6)         # [vx, vy, vz, wx, wy, wz]
        self.force = np.zeros(3)       # [fx, fy, fz]
        self.torque = np.zeros(3)      # [tx, ty, tz]
        self.q = np.zeros(7)           # 关节角度 [j1..j7]
        self.dq = np.zeros(7)          # 关节速度
        self.jacobian = np.zeros((6, 7))  # 雅可比矩阵 (6×7)

    def move_to_pose(self, pose: np.ndarray):
        """
        [REQUIRED] 发送末端笛卡尔位姿指令。

        Args:
            pose: shape (7,), [x, y, z, qx, qy, qz, qw]
                  - xyz 单位：米
                  - 四元数：基坐标系下，顺序 xyzw
                  - 已经过安全边界裁剪，可以直接执行

        控制模式：阻抗/导纳/位置伺服均可。
        Gym Env 侧已经做了 bounding box 裁剪，这里可以直接执行。
        """
        raise NotImplementedError

    def clear_error(self):
        """[REQUIRED] 清除错误状态，从 error 中恢复。"""
        raise NotImplementedError

    def joint_reset(self):
        """
        [REQUIRED 仅当使用 joint_reset_period]
        关节空间复位（比如跑一段时间后关节漂移需要回零）。
        如果不需要，可以留空。
        """
        raise NotImplementedError

    def update_control_params(self, params: dict):
        """
        [REQUIRED] 在线更新控制器参数。

        Gym Env 在 reset 和运行时频繁调用，用于切换柔顺/精密模式。
        例如 params = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            ...
        }

        如果你的控制器不支持动态调参，至少要兼容这个路由不报错。
        """
        raise NotImplementedError

    def read_state(self) -> dict:
        """
        [REQUIRED] 从硬件读回当前状态，更新内部缓存。
        在 /getstate 被调用前由路由函数调用（或者在路由函数内直接实现）。
        """
        raise NotImplementedError

# ============================================================================
# Flask 路由
# ============================================================================

def create_app(robot: RobotServer, gripper: GripperInterface) -> Flask:
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # [REQUIRED] 发送末端位姿指令
    # ------------------------------------------------------------------
    @app.route("/pose", methods=["POST"])
    def pose():
        """
        请求: {"arr": [x, y, z, qx, qy, qz, qw]}
        说明: 每步调用一次。env 已经用 ABS_POSE_LIMIT 做了安全裁剪。
        返回: 任意字符串
        """
        pos = np.array(request.json["arr"], dtype=np.float32)
        robot.move_to_pose(pos)
        return "Moved"

    # ------------------------------------------------------------------
    # [REQUIRED] 获取完整状态
    # ------------------------------------------------------------------
    @app.route("/getstate", methods=["POST"])
    def get_state():
        """
        每步调用一次，是 Gym Env 读取状态的唯一数据源。

        返回 JSON 结构（所有值都是 Python list 或 float）:
        {
            "pose":        [x, y, z, qx, qy, qz, qw],   # 7 个 float
            "vel":         [vx, vy, vz, wx, wy, wz],     # 6 个 float
            "force":       [fx, fy, fz],                 # 3 个 float
            "torque":      [tx, ty, tz],                 # 3 个 float
            "q":           [j1, j2, j3, j4, j5, j6, j7],# 7 个 float
            "dq":          [d1, d2, d3, d4, d5, d6, d7],# 7 个 float
            "jacobian":    [...],                        # 42 个 float (6×7 展平)
            "gripper_pos": 0.85                          # float, 0=闭合 1=全开
        }
        """
        robot.read_state()
        return jsonify({
            "pose":        robot.pose.tolist(),
            "vel":         robot.vel.tolist(),
            "force":       robot.force.tolist(),
            "torque":      robot.torque.tolist(),
            "q":           robot.q.tolist(),
            "dq":          robot.dq.tolist(),
            "jacobian":    robot.jacobian.flatten(order="F").tolist(),
            "gripper_pos": gripper.gripper_pos,
        })

    # ------------------------------------------------------------------
    # [REQUIRED] 清除错误
    # ------------------------------------------------------------------
    @app.route("/clearerr", methods=["POST"])
    def clearerr():
        """
        每步 _send_pos_command 和 _recover 都会调用。
        从 error state 恢复机械臂。
        """
        robot.clear_error()
        return "Clear"

    # ------------------------------------------------------------------
    # [REQUIRED] 夹爪 - 全开
    # ------------------------------------------------------------------
    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        """
        Gym Env 触发条件: action[6] >= 0.5 且 curr_gripper_pos < 0.85
        """
        gripper.open()
        return "Opened"

    # ------------------------------------------------------------------
    # [REQUIRED] 夹爪 - 全关
    # ------------------------------------------------------------------
    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        """
        Gym Env 触发条件: action[6] <= -0.5 且 curr_gripper_pos > 0.85
        """
        gripper.close()
        return "Closed"

    # ------------------------------------------------------------------
    # [REQUIRED 仅当任务有 regrasp 流程] 夹爪 - 慢速闭合
    # ------------------------------------------------------------------
    @app.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        """仅 RAM Insertion 等任务的 regrasp 流程调用。"""
        gripper.close_slow()
        return "Closed"

    # ------------------------------------------------------------------
    # [REQUIRED 仅当使用 joint_reset_period] 关节复位
    # ------------------------------------------------------------------
    @app.route("/jointreset", methods=["POST"])
    def jointreset():
        """
        触发: env 每 joint_reset_period 个 episode 调用一次。
        目的: 减少累积漂移。
        """
        robot.clear_error()
        robot.joint_reset()
        return "Reset Joint"

    # ------------------------------------------------------------------
    # [REQUIRED] 更新控制器参数
    # ------------------------------------------------------------------
    @app.route("/update_param", methods=["POST"])
    def update_param():
        """
        请求: COMPLIANCE_PARAM 或 PRECISION_PARAM 字典。
        训练时用柔顺模式（COMPLIANCE），复位时用精密模式（PRECISION）。

        如果控制器不支持动态调参，至少不要报错，
        让训练能正常进行。
        """
        robot.update_control_params(request.json)
        return "Updated compliance parameters"

    # ------------------------------------------------------------------
    # [OPTIONAL] 以下端点仅用于人类调试，Gym Env 从不调用
    # ------------------------------------------------------------------

    @app.route("/getpos", methods=["POST"])
    def get_pos():
        """返回末端位姿 xyz + 四元数。调试用。"""
        return jsonify({"pose": robot.pose.tolist()})

    @app.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        """返回末端位姿 xyz + 欧拉角。调试用。"""
        # 需要 scipy: from scipy.spatial.transform import Rotation as R
        # xyz = robot.pose[:3]
        # r = R.from_quat(robot.pose[3:]).as_euler("xyz")
        # return jsonify({"pose": np.concatenate([xyz, r]).tolist()})
        return jsonify({"pose": robot.pose.tolist()})  # 简化版

    @app.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": robot.vel.tolist()})

    @app.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": robot.force.tolist()})

    @app.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": robot.torque.tolist()})

    @app.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": robot.q.tolist()})

    @app.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": robot.dq.tolist()})

    @app.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": robot.jacobian.flatten(order="F").tolist()})

    @app.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper.gripper_pos})

    @app.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        return "Activated"

    @app.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        return "Reset"

    @app.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = int(request.json["gripper_pos"])
        print(f"move gripper to {pos}")
        return "Moved Gripper"

    @app.route("/startimp", methods=["POST"])
    def start_impedance():
        return "Started impedance"

    @app.route("/stopimp", methods=["POST"])
    def stop_impedance():
        return "Stopped impedance"

    return app


# ============================================================================
# 启动入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Flask 绑定地址")
    parser.add_argument("--port", default=5000, type=int, help="Flask 端口")
    args = parser.parse_args()

    robot = RobotServer()
    gripper = GripperInterface()

    app = create_app(robot, gripper)
    app.run(host=args.host, port=args.port)

