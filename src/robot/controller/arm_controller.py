import numpy as np
from typing import Dict, Any
from robot.controller.controller import Controller
from robot.utils.base.data_handler import debug_print

# 基于Controller基类的机械臂控制器类，包含机械臂特有的状态信息和控制方法
# 需要子类实现
# get_state() 获取机械臂状态的方法，返回一个包含机械臂状态信息的字典
# set_position(position) 设置机械臂末端位置的方法，接受一个包含位置坐标的数组，位置坐标可以是欧拉角或者四元数
# set_joint(joint) 设置机械臂关节位置的方法，接受一个包含关节位置的数组
# set_gripper(gripper) 设置机械臂夹爪状态的方法，接受一个表示夹爪开合程度的数值，范围在0到1之间
# ...
class ArmController(Controller):
    def __init__(self):
        super().__init__()
        self.name = "arm_controller"
        self.controller_type = "robotic_arm"
        self.controller = None
    # 子类需要实现的机械臂状态获取方法，返回一个包含机械臂状态信息的字典
    def get_information(self):
        arm_info = {}
        state = self.get_state()
        if "joint" in self.collect_info:
            arm_info["joint"] = state["joint"]
        if "eef" in self.collect_info:
            arm_info["eef"] = state["eef"]
        if "gripper" in self.collect_info:
            arm_info["gripper"] = state["gripper"]
        return arm_info
    # 子类需要实现的机械臂移动方法，接受一个包含移动信息的字典和一个表示是否为增量的布尔值
    def move_controller(self, move_data:Dict[str, Any], is_delta=False):
        if is_delta:
            now_state = self.get_state()
            for key, value in move_data.items():
                if key == "joint":
                    self.set_joint(np.array(now_state["joint"] + value))
                elif key == "eef":
                    self.set_position(np.array(now_state["eef"] + value))
        else:
            for key, value in move_data.items():
                if key == "joint":
                    self.set_joint(np.array(value))
                elif key == "eef":
                    self.set_position(np.array(value))
        
        # For action and gripper, use absolute values instead of deltas
        for key, value in move_data.items():
            if key == "teleop_qpos":
                self.set_position_teleop(np.array(value))
            if key == "action":
                self.set_action(np.array(value))
            if key == "gripper":
                self.set_gripper(np.array(value))
            if key == "velocity":
                self.set_velocity(np.array(value))
            if key == "force":
                self.set_force(np.array(value))

    def __repr__(self):
        if self.controller is not None:
            return f"{self.name}: \n \
                    controller: {self.controller}"
        else:
            return super().__repr__()