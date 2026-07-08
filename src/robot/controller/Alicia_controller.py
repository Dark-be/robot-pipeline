
import os
import time

import numpy as np

from robot.controller.arm_controller import ArmController
from robot.utils.base.data_handler import debug_print

import alicia_d_sdk

class AliciaController(ArmController):
	"""Alicia 示教臂 Controller。

	- 读取：完全参考 third_party/Alicia-D-SDK/examples/03_demo_read_state.py
	  使用 robot.get_robot_state("joint_gripper") 获取 state.angles
	- 控制：仅支持关节角度（6 DoF, 弧度）下发，不做任何 IK / pose 解算
	"""

	def __init__(self, name: str, INFO: str = "INFO"):
		super().__init__()
		self.name = name
		self.controller_type = "user_controller"
		self.INFO = INFO

		self.robot = None
		self.port: str = ""

	def set_up(self, port: str):
		"""初始化连接机器人，设置串口和夹爪型号等参数。
		:param port: 串口端口，如 /dev/ttyACM0
		"""
		self.port = port

		self.robot = alicia_d_sdk.create_robot(
			port=port,
			gripper_type="50mm",
			debug_mode=False,
			auto_connect=True,
		)

		debug_print(self.name, f"setup success, port={port}", self.INFO)

	def disconnect(self):
		if self.robot is None:
			return
		try:
			self.robot.disconnect()
		finally:
			self.robot = None

	def get_state(self):
		"""返回 state dict 至少包含 joint/gripper/eef key。

		joint: np.ndarray shape (6,), rad
		gripper: float in [0, 1] (SDK 原始值 0-1000 归一化)
		eef: None 不提供 pose/解算
		"""
		if self.robot is None:
			raise RuntimeError(f"{self.name}: controller is not set up (robot is None)")
		state = self.robot.get_robot_state("joint_gripper")
		if state is None:
			return {"joint": None, "gripper": None, "eef": None}
		offset = [0, -2.0, 0.4, 0, 0.6, 0]  # SDK 角度偏移（弧度），需要根据实际情况调整
		multiplier = [1, -1.3, -1, -1, -1, -1]
		joint = np.asarray(state.angles, dtype=float)
		joint = np.array([(joint[i] + offset[i]) * multiplier[i] for i in range(6)], dtype=float)
		gripper_raw = float(state.gripper)  # SDK: 0-1000
		gripper = max(0.0, min(1.0, gripper_raw / 1000.0))
		
		return {"joint": joint, "gripper": gripper, "eef": None}
	
	def set_joint(
		self,
		joint: np.ndarray,
		speed_deg_s: float = 10,
		wait_for_completion: bool = False,
	):
		"""下发关节角度（弧度）。"""
		if self.robot is None:
			raise RuntimeError(f"{self.name}: controller is not set up (robot is None)")

		joint_arr = np.asarray(joint, dtype=float).reshape(-1)
		if joint_arr.shape[0] != 6:
			raise ValueError(f"{self.name}: joint should have 6 elements, got {joint_arr.shape[0]}")

		ok = self.robot.set_robot_state(
			target_joints=joint_arr.tolist(),
			gripper_value=None,
			joint_format="rad",
			speed_deg_s=float(speed_deg_s),
			wait_for_completion=bool(wait_for_completion),
		)
		if not ok:
			debug_print(self.name, "set_joint failed", "WARNING")

	def __del__(self):
		try:
			self.disconnect()
		except Exception:
			pass

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Alicia controller (read joint + set joint only)")
	parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="串口端口 (例如: /dev/ttyACM0)")
	parser.add_argument("--gripper_type", type=str, default="50mm", help="夹爪型号 (默认: 50mm)")
	parser.add_argument("--fps", type=float, default=30.0, help="读取频率 (Hz)")
	parser.add_argument("--speed", type=float, default=10.0, help="关节速度 (deg/s)")
	parser.add_argument(
		"--set_joint",
		nargs=6,
		metavar=("j1", "j2", "j3", "j4", "j5", "j6"),
		help="下发 6 关节角度（弧度），例如：--set_joint 0 0 0 0 0 0",
	)

	args = parser.parse_args()

	os.environ.setdefault("INFO_LEVEL", "INFO")

	ctrl = AliciaController("alicia")
	ctrl.set_collect_info(["joint", "gripper"])  # 不采集 eef
	ctrl.set_up(port=args.port, gripper_type=args.gripper_type)

	try:
		if True:
			# target = _parse_floats(args.set_joint)
			target = np.asarray([0, 0, 0.78, 0,-1.57, 0], dtype=float)
			ctrl.set_joint(target, speed_deg_s=args.speed, wait_for_completion=False)
			time.sleep(4)
			ctrl.robot.torque_control('off')

		interval = 1.0 / max(1e-6, float(args.fps))
		while True:
			s = ctrl.get_state()
			j = s["joint"]
			g = s["gripper"]
			if j is not None:
				print("joint(rad):", " ".join(f"{x:.3f}" for x in j.tolist()), "gripper:", f"{g:.3f}" if g is not None else "None")
			else:
				print("joint: None, gripper: None")
			time.sleep(interval)
	except KeyboardInterrupt:
		pass
	finally:
		ctrl.disconnect()
