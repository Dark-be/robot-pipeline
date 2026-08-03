import numpy as np
from controller.arm_controller import ArmController
from utils.base.data_handler import debug_print

# 基于ArmController的测试机械臂控制器类，包含机械臂状态获取和控制方法的简单实现，用于测试和调试
class TestArmController(ArmController):
    def __init__(self, name="test_arm"):
        super().__init__(name)
    
    def connect(self):
        debug_print(self.name, f"setup success", "INFO")

    def disconnect(self):
        debug_print(self.name, f"disconnect success", "INFO")

    def get_state(self):
        state = {}
        state["joint"] = np.random.rand(self.DoFs) * 3.1515926 if self.state == {} or "joint" not in self.state.keys() \
              else self.state["joint"]
        state["gripper"] = np.random.rand(1) if self.state == {}  or "gripper" not in self.state.keys() \
              else self.state["gripper"]
        state["pose"] = np.random.rand(6) if self.state == {}  or "pose" not in self.state.keys() \
              else self.state["pose"]
        return state

    def get_joint(self):
        return self.get_state()["joint"]

    def get_position(self):
        return self.get_state()["pose"]

    def get_gripper(self):
        return self.get_state()["gripper"]

    def set_position(self, position: np.ndarray):
        if position.shape[0] == 6:
            debug_print(self.name, f"using EULER set position to {position}", "DEBUG")
        elif position.shape[0] == 7:
            debug_print(self.name, f"using QUATERNION set position to {position}", "DEBUG")
        else:
            debug_print(self.name, f"set_position input size should be 6 -> EULER or 7 -> QUATERNION","ERROR")
        
        self.state["pose"] = position
    
    def set_joint(self, joint: np.ndarray):
        debug_print(self.name, f"set joint to {joint}", "DEBUG")
        
        self.state["joint"] = joint

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper: float):
        if isinstance(gripper, (int, float, complex,np.ndarray)) and not isinstance(gripper, bool):
            if 1 >= gripper >= 0:
                debug_print(self.name, f"set gripper to {gripper}", "INFO")
            else:
                debug_print(self.name, f"gripper better be 0~1, but get number {gripper}","WARNING")
        else:
            debug_print(self.name, f"gripper should be a number 0~1, but get type {type(gripper)}","ERROR")
        
        self.state["gripper"] = gripper