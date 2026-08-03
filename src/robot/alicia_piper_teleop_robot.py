import time
import numpy as np
from robot.base_robot import Robot
from controller.alicia_teach_controller import AliciaTeachController
from controller.piper_controller import PiperController

from sensor.v4l2_sensor import V4l2Sensor
from sensor.realsense_sensor import RealsenseSensor

from utils.base.data_handler import debug_print

import rerun as rr

class Alicia_Piper_Teleop_Robot(Robot):

    def __init__(self, robot_config: dict):
        super().__init__(robot_config=robot_config)

        self.controllers = {
            "master_left_arm": AliciaTeachController("master_left_arm"),
            "master_right_arm": AliciaTeachController("master_right_arm"),
            "slave_left_arm": PiperController("slave_left_arm"),
            "slave_right_arm": PiperController("slave_right_arm"),
        }
        self.sensors = {
            "cam_head": RealsenseSensor("cam_head"),
            "cam_left_wrist": V4l2Sensor("cam_left_wrist"),
            "cam_right_wrist": V4l2Sensor("cam_right_wrist"),
        }

        # reset()/sync() 限步插值共用的从臂命令位置（前馈插值的当前值）
        self._slave_left_joint_cmd = None
        self._slave_right_joint_cmd = None

    def connect(self):
        master_port_left = self.robot_config["MASTER_PORT"]["left_arm"]
        master_port_right = self.robot_config["MASTER_PORT"]["right_arm"]
        slave_port_left = self.robot_config["SLAVE_PORT"]["left_arm"]
        slave_port_right = self.robot_config["SLAVE_PORT"]["right_arm"]
        if not master_port_left:
            debug_print(self.name, "MASTER_PORT for left_arm is not configured. Please check your config.yml", "ERROR")
        if not master_port_right:
            debug_print(self.name, "MASTER_PORT for right_arm is not configured. Please check your config.yml", "ERROR")
        if not slave_port_left:
            debug_print(self.name, "SLAVE_PORT for left_arm is not configured. Please check your config.yml", "ERROR")
        if not slave_port_right:
            debug_print(self.name, "SLAVE_PORT for right_arm is not configured. Please check your config.yml", "ERROR")

        head_cam_serial = self.robot_config["CAMERA_SERIALS"]["head"]
        left_cam_serial = self.robot_config["CAMERA_SERIALS"]["left"]
        right_cam_serial = self.robot_config["CAMERA_SERIALS"]["right"]
        if not head_cam_serial:
            debug_print(self.name, "CAMERA_SERIALS for head is not configured. Please check your config.yml", "ERROR")
        if not left_cam_serial:
            debug_print(self.name, "CAMERA_SERIALS for left is not configured. Please check your config.yml", "ERROR")
        if not right_cam_serial:
            debug_print(self.name, "CAMERA_SERIALS for right is not configured. Please check your config.yml", "ERROR")

        self.controllers["master_left_arm"].connect(port=master_port_left)
        self.controllers["master_left_arm"].set_collect_info(["joint", "gripper"])
        
        self.controllers["master_right_arm"].connect(port=master_port_right)
        self.controllers["master_right_arm"].set_collect_info(["joint", "gripper"])

        self.controllers["slave_left_arm"].connect(port=slave_port_left)
        self.controllers["slave_left_arm"].set_collect_info(["joint", "gripper"])

        self.controllers["slave_right_arm"].connect(port=slave_port_right)
        self.controllers["slave_right_arm"].set_collect_info(["joint", "gripper"])

        self.sensors["cam_head"].connect(device=head_cam_serial, is_jpeg=True)
        self.sensors["cam_head"].set_collect_info(["color"])

        self.sensors["cam_left_wrist"].connect(device=left_cam_serial, is_jpeg=True)
        self.sensors["cam_left_wrist"].set_collect_info(["color"])

        self.sensors["cam_right_wrist"].connect(device=right_cam_serial, is_jpeg=True)
        self.sensors["cam_right_wrist"].set_collect_info(["color"])
        
        debug_print(self.name, f"Setup complete.", "INFO")
        time.sleep(1)
        self.ready = True

    def get_standard_obs(self):
        obs = self.get_obs()
        controller_data, sensor_data = obs
        standard_obs = {
            "observations/qpos": np.concatenate([
                np.asarray(controller_data["slave_left_arm"]["joint"]).ravel(),
                np.asarray(controller_data["slave_left_arm"]["gripper"]).ravel(),
                np.asarray(controller_data["slave_right_arm"]["joint"]).ravel(),
                np.asarray(controller_data["slave_right_arm"]["gripper"]).ravel(),
            ]),
            "observations/images/cam_head": sensor_data["cam_head"]["color"],
            "observations/images/cam_left_wrist": sensor_data["cam_left_wrist"]["color"],
            "observations/images/cam_right_wrist": sensor_data["cam_right_wrist"]["color"],
            "action": np.concatenate([
                self._slave_left_joint_cmd,
                np.asarray(controller_data["master_left_arm"]["gripper"]).ravel(),
                self._slave_left_joint_cmd,
                np.asarray(controller_data["master_right_arm"]["gripper"]).ravel(),
            ])
        }

        return standard_obs

    def _step_toward(self, current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        delta = target - current
        step = np.clip(delta, -max_step, max_step)
        # 距离 <= max_step 的关节直接取目标值，避免浮点累加误差，便于精确判定收敛
        return np.where(np.abs(delta) <= max_step, target, current + step)

    def _read_slave_joint_state(self):
        """读取两条从臂当前的实际关节角度（弧度），同时更新 robot 内部状态（controller_data）。"""
        obs = self.get_controller_obs()
        left_data = obs.get("slave_left_arm") or {}
        right_data = obs.get("slave_right_arm") or {}
        left = left_data.get("joint")
        right = right_data.get("joint")
        if left is None or right is None:
            raise RuntimeError("Failed to read slave arm joint states during reset.")
        return np.asarray(left, dtype=np.float64).ravel(), np.asarray(right, dtype=np.float64).ravel()

    def reset(self):
        init_qpos = self.robot_config.get("init_qpos", {})
        left_init = np.asarray(init_qpos.get("slave_left_arm"), dtype=np.float64)
        right_init = np.asarray(init_qpos.get("slave_right_arm"), dtype=np.float64)

        left_target_joint = left_init[:6]
        right_target_joint = right_init[:6]
        left_target_gripper = float(left_init[6])
        right_target_gripper = float(right_init[6])

        step_rad = float(self.robot_config.get("reset_step_rad", 0.05))          # 单步最大关节增量(rad)
        interval = float(self.robot_config.get("reset_interval", 1.0 / 30.0))    # 相邻两步命令的时间间隔(秒)，约 30Hz
        max_time = float(self.robot_config.get("reset_max_time", 5.0))          # 复位超时保护(秒)

        max_step = step_rad

        # 1. 读取一次当前实际关节角度作为插值起点（同时更新 robot 内部状态），写入成员命令位置
        self._slave_left_joint_cmd, self._slave_right_joint_cmd = self._read_slave_joint_state()
        debug_print(self.name, f"Reseting slave arms, max speed: {max_step} rad/step", "INFO")
        # 2. 基于命令位置做前馈插值：每步向目标靠近 0.05 rad，循环中不再回读实际状态。
        start_time = time.time()
        while True:
            self._slave_left_joint_cmd = self._step_toward(self._slave_left_joint_cmd, left_target_joint, max_step)
            self._slave_right_joint_cmd = self._step_toward(self._slave_right_joint_cmd, right_target_joint, max_step)

            action = {
                "slave_left_arm": {
                    "joint": self._slave_left_joint_cmd,
                    "gripper": left_target_gripper,
                },
                "slave_right_arm": {
                    "joint": self._slave_right_joint_cmd,
                    "gripper": right_target_gripper,
                }
            }
            self.take_action(action)

            # 命令位置作为下一步的起点（前馈插值，成员变量已在循环内更新）
            if np.all(self._slave_left_joint_cmd == left_target_joint) and \
               np.all(self._slave_right_joint_cmd == right_target_joint):
                debug_print(self.name, "Reset: reached target configuration.", "INFO")
                break

            if time.time() - start_time > max_time:
                debug_print(self.name, "Reset: interpolation timed out. "
                           f"Remaining L={np.round(left_target_joint - self._slave_left_joint_cmd, 4)} "
                           f"R={np.round(right_target_joint - self._slave_right_joint_cmd, 4)}", "WARNING")
                break

            time.sleep(interval)

        # 3. 确保最终命令位置精确到目标值，避免浮点累加误差
        final_action = {
            "slave_left_arm": {
                "joint": left_target_joint,
                "gripper": left_target_gripper,
            },
            "slave_right_arm": {
                "joint": right_target_joint,
                "gripper": right_target_gripper,
            }
        }
        self.take_action(final_action)

    def visualize(self):
        controller_data = self.controller_data
        for arm_name in ["slave_left_arm", "slave_right_arm"]:
            arm_data = controller_data.get(arm_name)
            if not arm_data:
                debug_print(self.name, f"No data received for {arm_name}. Skipping visualization.", "WARNING")
                continue
            joint_angles = arm_data.get("joint")
            gripper_pos = arm_data.get("gripper")


            rr.log(f"{arm_name}/qpos", rr.Scalars(np.asarray(joint_angles).ravel()))

        rr.log("action_left", rr.Scalars(np.asarray(self._slave_left_joint_cmd).ravel()))
        rr.log("action_right", rr.Scalars(np.asarray(self._slave_right_joint_cmd).ravel()))

        sensor_data = self.sensor_data

        for cam_name, cam_data in sensor_data.items():
            img_bytes = cam_data.get("color")
            if img_bytes is None:
                debug_print(self.name, f"No color image data for {cam_name}. Skipping visualization.", "WARNING")
                continue
            rr.log(f"images/{cam_name}", rr.EncodedImage(contents=img_bytes, media_type="image/jpeg"))


    def sync(self):
        master_left_data = self.controller_data.get("master_left_arm")
        master_right_data = self.controller_data.get("master_right_arm")
        if not master_left_data or not master_right_data:
            debug_print(self.name, "No data received from master controllers. Skipping sync.", "WARNING")
            return
        
        joint_left = master_left_data.get("joint")
        joint_right = master_right_data.get("joint")
        if joint_left is None or joint_right is None:
            debug_print(self.name, "Joint data from master controllers is missing. Skipping sync.", "WARNING")
            return

        gripper_left = master_left_data.get("gripper")
        gripper_right = master_right_data.get("gripper")
        if gripper_left is None or gripper_right is None:
            debug_print(self.name, "Gripper data from master controllers is missing. Skipping sync.", "WARNING")
            return

        # 跟随步长：每帧最多向主手目标靠近 sync_step_rad（rad），防止从臂快速跳变超出限位
        step_rad = float(self.robot_config.get("sync_step_rad", 0.05))

        # 首帧以当前从臂实际位置作为跟随起点，避免一开始就跳变
        if self._slave_left_joint_cmd is None or self._slave_right_joint_cmd is None:
            try:
                self._slave_left_joint_cmd, self._slave_right_joint_cmd = self._read_slave_joint_state()
            except RuntimeError as e:
                debug_print(self.name, f"Failed to init sync command from slave joints: {e}", "WARNING")
                return

        left_cmd = self._step_toward(self._slave_left_joint_cmd, joint_left, step_rad)
        right_cmd = self._step_toward(self._slave_right_joint_cmd, joint_right, step_rad)
        self._slave_left_joint_cmd = left_cmd
        self._slave_right_joint_cmd = right_cmd

        self.take_action({
            "slave_left_arm": {
                "joint": left_cmd,
                "gripper": float(gripper_left),
            },
            "slave_right_arm": {
                "joint": right_cmd,
                "gripper": float(gripper_right),
            }
        })

        return