import numpy as np

from controller.arm_controller import ArmController
from utils.base.data_handler import debug_print

import alicia_d_sdk

import robocore as rc
from robocore.kinematics import forward_kinematics
from robocore.transform import matrix_to_euler, matrix_to_quaternion

class AliciaController(ArmController):
	def __init__(self, name="alicia_controller"):
		super().__init__(name)
		self.robot = None
		self.port: str = "/dev/ttyACM0"
		# rc.set_backend("numpy")

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
		offset = [0, -2.0, 0.4, 0, 0.9, 0]  # SDK 角度偏移（弧度），需要根据实际情况调整
		multiplier = [1, -1.3, -1, -1, -1, -1]

		joint = np.asarray(state.angles, dtype=float)
		joint = np.array([(joint[i] + offset[i]) * multiplier[i] for i in range(6)], dtype=float)
		gripper_raw = float(state.gripper)  # SDK: 0-1000
		gripper = max(0.0, min(1.0, gripper_raw / 1000.0))

		T_fk = forward_kinematics(
      self.robot.robot_model,
      state.angles,
      return_end=True
    )
		position_fk = T_fk[:3, 3]
		rotation_fk = T_fk[:3, :3]
		euler_fk = matrix_to_euler(rotation_fk)
		return {"joint": joint, "gripper": gripper, "pose": np.append(position_fk, euler_fk)}  # 返回机械臂状态信息，包括关节角度、夹爪状态和末端位姿

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
	
