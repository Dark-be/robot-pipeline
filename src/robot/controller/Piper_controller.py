import numpy as np
import time
from robot.controller.arm_controller import ArmController
from robot.utils.base.data_handler import debug_print
from robot.utils.kenimatics import get_tool_position
from piper_sdk import *

# 基于ArmController的测试机械臂控制器类，包含机械臂状态获取和控制方法的简单实现，用于测试和调试
class PiperController(ArmController):
  def __init__(self, name, INFO = "INFO"):
    super().__init__()
    self.name = name
    self.controller_type = "user_controller"
    self.INFO = INFO

    self.piper = None
    self.port: str = "can0"
  
  def set_up(self, port: str = "can0"):
    self.port = port
    self.piper = C_PiperInterface_V2(
      can_name=port,
      judge_flag=False,
      can_auto_init=True,
      dh_is_offset=1,
      start_sdk_joint_limit=True,
      start_sdk_gripper_limit=False,
      logger_level=LogLevel.WARNING,
      log_to_file=False,
      log_file_path=None)
    self.piper.ConnectPort()

    while( not self.piper.EnablePiper()):
      time.sleep(0.01)
    self.piper.MotionCtrl_2(0x01, 0x01, 40, 0x00)

  def disconnect(self):
    if self.piper is not None:
      self.piper.DisconnectPort()
      debug_print(self.name, f"Disconnected from Piper on port {self.port}", self.INFO)
      self.piper = None

  def get_state(self):
    if self.piper is None:
        raise RuntimeError("PiperController not set up. Call set_up() before get_state().")
    
    joint_msgs = self.piper.GetArmJointMsgs().joint_state
    gripper_msgs = self.piper.GetArmGripperMsgs().gripper_state
    state = {}
    # randly return a vaild value  
    if joint_msgs is None or gripper_msgs is None:
      debug_print(self.name, f"Failed to get joint or gripper messages from Piper SDK")
      return {"joint": None, "gripper": None, "eef": None}
    state["joint"] = np.array([
      joint_msgs.joint_1,
      joint_msgs.joint_2,
      joint_msgs.joint_3,
      joint_msgs.joint_4,
      joint_msgs.joint_5,
      joint_msgs.joint_6
    ]) / 1000.0 / 180.0 * 3.14159  # Convert to radians

    state["gripper"] = gripper_msgs.grippers_angle / 100000.0

    
    state["eef"] = [0,0,0,0,0,0]

    return state
  
  def set_joint(
    self, 
    joint: np.ndarray
  ):
    if self.piper is None:
      raise RuntimeError("PiperController not set up. Call set_up() before set_joint().")
    if joint.shape[0] != 6:
      debug_print(self.name, f"set_joint() input size should be 6","ERROR")   
    else: 
      # Convert radians to Piper SDK units (0-1000 for 0-180 degrees)
      joint_cmd = (joint / 3.14159 * 180.0 * 1000).astype(int)
      self.piper.JointCtrl(
        joint_1=joint_cmd[0],
        joint_2=joint_cmd[1],
        joint_3=joint_cmd[2],
        joint_4=joint_cmd[3],
        joint_5=joint_cmd[4],
        joint_6=joint_cmd[5]
      )

      debug_print(self.name, f"set joint to {joint}", self.INFO)

  # The input gripper value is in the range [0, 1], representing the degree of opening.
  def set_gripper(self, gripper):
    if self.piper is None:
      raise RuntimeError("PiperController not set up. Call set_up() before set_gripper().")
    if not isinstance(gripper, (int, float, complex, np.ndarray)) or isinstance(gripper, bool):
      debug_print(self.name, f"gripper should be a number 0~1, but get type {type(gripper)}", "ERROR")
    elif not (1 >= gripper >= 0):
      debug_print(self.name, f"gripper better be 0~1, but get number {gripper}", "WARNING")
    else:
      gripper_cmd = int(gripper * 100000)  # Convert to Piper SDK
      self.piper.GripperCtrl(
        gripper_angle=gripper_cmd,
        gripper_effort=1000,  #(0-5000)
        gripper_code=0x01  # 0x01 enable
      )
      debug_print(self.name, f"set gripper to {gripper}", self.INFO)

if __name__=="__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"
    
    controller = PiperController("piper_controller", INFO="DEBUG")
    controller.set_up(port="can_left")

    state = controller.get_state()
    print(f"Initial state: {state}")
    
    controller.set_gripper(0.99)
    for i in range(40):
        controller.set_joint(np.array([0.036, 0.046, -0.407, -0.081, 0.471, 0.216]))
        time.sleep(0.05)  # Wait for the commands to take effect

    controller.set_joint(np.array([0.036, 0.046, -0.407, -0.081, 0.471, 0.216]))
    time.sleep(1)  # Wait for the commands to take effect
    state = controller.get_state()
    print(f"State after commands: {state}")