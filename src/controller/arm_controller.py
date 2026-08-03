# 基于Controller基类的机械臂控制器类，包含机械臂特有的状态信息和控制方法
# 需要子类实现
# get_state() 获取机械臂状态的方法，返回一个包含机械臂状态信息的字典
# set_position(position) 设置机械臂末端位置的方法，接受一个包含位置坐标的数组，位置坐标可以是欧拉角或者四元数
# set_joint(joint) 设置机械臂关节位置的方法，接受一个包含关节位置的数组
# set_gripper(gripper) 设置机械臂夹爪状态的方法，接受一个表示夹爪开合程度的数值，范围在0到1之间
# ...
import numpy as np
from controller.controller import Controller
from utils.base.data_handler import debug_print
class ArmController(Controller):
    def __init__(self, name="arm_controller"):
        super().__init__(name)
        self.state = {"joint": None, "gripper": None, "pose": None}  # 机械臂状态信息，包含关节位置、末端位置和夹爪状态等

    def get_information(self):
        if self.collect_info is None:
            debug_print(self.name, f"collect_info is not set", "WARNING")
            return None
        
        if "joint" in self.collect_info:
            self.state["joint"] = self.get_joint()
        if "gripper" in self.collect_info:
            self.state["gripper"] = self.get_gripper()
        if "pose" in self.collect_info:
            self.state["pose"] = self.get_position()
        return self.state.copy()  # 返回机械臂状态信息的副本，避免外部修改原始状态
    
    def take_action(self, action: dict):
        for key, value in action.items():
            if key == "joint":
                self.set_joint(np.array(value))
            elif key == "gripper":
                self.set_gripper(value)
            elif key == "pose":
                self.set_position(np.array(value))

    def get_joint(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_position(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_gripper(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_joint(self, joint: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_position(self, position: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_gripper(self, gripper: float):
        raise NotImplementedError("Subclasses should implement this method.")