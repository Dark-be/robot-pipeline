import numpy as np
import time
from robot.controller.arm_controller import ArmController
from robot.utils.base.data_handler import debug_print
from robot.utils.kenimatics import get_tool_position
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

MIT_CTRL_CFG = [
  { "vel_ref": 0.0, "kp": 3.0, "kd": 0.7, "t_ref": 0.0},
  { "vel_ref": 0.0, "kp": 3.0, "kd": 0.8, "t_ref": 0.0},
  { "vel_ref": 0.0, "kp": 3.0, "kd": 0.8, "t_ref": 0.0},
  { "vel_ref": 0.0, "kp": 3.0, "kd": 0.6, "t_ref": 0.0},
  { "vel_ref": 0.0, "kp": 3.0, "kd": 0.6, "t_ref": 0.0},
  { "vel_ref": 0.0, "kp": 2.0, "kd": 0.5, "t_ref": 0.0},
]

# 基于ArmController的测试机械臂控制器类，包含机械臂状态获取和控制方法的简单实现，用于测试和调试
class PiperController(ArmController):
  def __init__(self, name):
    super().__init__()
    self.name = name
    self.controller_type = "user_controller"

    self.piper = None
    self.gripper = None
    self.port: str = "can0"
    self.ctrl_mode: str = "joint"  # 控制模式，支持 "joint" 或 "pose"
  
  def set_up(self, port: str = "can0", ctrl_mode: str = "mit"):
    self.port = port
    self.ctrl_mode = ctrl_mode

    cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V188, channel=port)
    self.piper = AgxArmFactory.create_arm(cfg)
    self.piper.set_joint_limits_enabled(True)
    self.gripper = self.piper.init_effector(self.piper.OPTIONS.EFFECTOR.AGX_GRIPPER)
    self.piper.connect(start_read_thread=True)

    while(not self.piper.enable()):
      time.sleep(0.01)

  def disconnect(self):
    if self.piper is not None:
      self.piper.disconnect()
      debug_print(self.name, f"Disconnected from Piper on port {self.port}", "INFO")
      self.piper = None

  def get_state(self):
    if self.piper is None:
        raise RuntimeError("PiperController not set up. Call set_up() before get_state().")
    
    state = {}
    joint_angles = self.piper.get_joint_angles()
    gripper_state = self.gripper.get_gripper_status()
    flange_pose = self.piper.get_flange_pose()
  
    if joint_angles is None:
      debug_print(self.name, f"Failed to get joint angles")
      return {"joint": None, "gripper": None, "eef": None}
    state["joint"] = np.array(joint_angles.msg)

    state["gripper"] = gripper_state.msg.value * 10

    state["eef"] = np.array(flange_pose.msg)

    return state
  
  def set_joint(self, joint: np.array):
    if self.piper is None:
      raise RuntimeError("PiperController not set up. Call set_up() before set_joint().")
    
    if joint.shape[0] != 6:
      debug_print(self.name, f"set_joint() input size should be 6","ERROR")   
    else: 
      joint[2] = np.clip(joint[2], -2.96706, 0.0)
      if self.ctrl_mode == "mit":
        for i in range(6):
          self.piper.move_mit(
            joint_index = i + 1,
            p_des = joint[i],
            v_des = MIT_CTRL_CFG[i]["vel_ref"],
            kp = MIT_CTRL_CFG[i]["kp"],
            kd = MIT_CTRL_CFG[i]["kd"],
            t_ff = MIT_CTRL_CFG[i]["t_ref"]
          )

      elif self.ctrl_mode == "joint":
        self.piper.move_j(joint.tolist())

      debug_print(self.name, f"set joint to {joint}", "DEBUG")

  def set_pose(self, pose: np.array):
    self.piper.move_p(pose.tolist())

  def set_gripper(self, gripper: float):
    if self.piper is None:
      raise RuntimeError("PiperController not set up. Call set_up() before set_gripper().")
    elif not (1 >= gripper >= 0):
      gripper = np.clip(gripper, 0, 1)
      debug_print(self.name, f"gripper better be 0~1, but get number {gripper}", "WARNING")
    else:
      gripper_cmd = gripper / 10
      self.gripper.move_gripper_m(gripper_cmd)
      debug_print(self.name, f"set gripper to {gripper}", "DEBUG")

if __name__=="__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"
    
    controller = PiperController("piper_controller")
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