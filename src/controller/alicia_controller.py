import time

import numpy as np

from controller.arm_controller import ArmController
from utils.base.data_handler import debug_print
from robot.types import CONTROLLER_TYPE

import alicia_d_sdk

from alicia_d_sdk.utils.logger import BeautyLogger, LogLevel

# 只显示 ERROR 级别的日志
logger = BeautyLogger(
    log_dir="logs", 
    log_name="app.log", 
    min_level=LogLevel.ERROR
)

class AliciaController(ArmController):
	def __init__(self, name="alicia_controller"):
		super().__init__(name)
		self.robot = None
		self.port: str = "/dev/ttyACM0"

	def connect(self, port: str):
		self.port = port

		self.robot = alicia_d_sdk.create_robot(
			port=port,
			gripper_type="50mm",
			debug_mode=False,
			auto_connect=True,
		)
		self.robot.torque_control('off')
		debug_print(self.name, f"Connected to Alicia on port {port}", "INFO")

	def disconnect(self):
		if self.robot is None:
			return
		try:
			self.robot.disconnect()
		finally:
			self.robot = None

	def get_state(self):
		if self.robot is None:
			raise RuntimeError(f"{self.name}: controller is not set up (robot is None)")

		state = self.robot.get_robot_state("joint_gripper")
		if state is None:
			return {"joint": None, "gripper": None, "pose": None}
		offset = [0, -2.0, 0.4, 0, 0.6, 0]  # SDK 角度偏移（弧度），需要根据实际情况调整
		multiplier = [1, -1.3, -1, -1, -1, -1]

		joint = np.asarray(state.angles, dtype=float)
		joint = np.array([(joint[i] + offset[i]) * multiplier[i] for i in range(6)], dtype=float)
		gripper_raw = float(state.gripper)  # SDK: 0-1000
		gripper = max(0.0, min(1.0, gripper_raw / 1000.0))
		
		return {"joint": joint, "gripper": gripper}

	def get_information(self):
		if self.collect_info is None:
			debug_print(self.name, f"collect_info is not set", "WARNING")
			return None
		
		state = self.get_state()
		if "joint" in self.collect_info:
			self.state["joint"] = state["joint"]
		if "gripper" in self.collect_info:
			self.state["gripper"] = state["gripper"]
		return self.state.copy()  # 返回机械臂状态信息的副本，避免外部修改原始状态
	
