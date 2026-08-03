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
                np.asarray(controller_data["master_left_arm"]["joint"]).ravel(),
                np.asarray(controller_data["master_left_arm"]["gripper"]).ravel(),
                np.asarray(controller_data["master_right_arm"]["joint"]).ravel(),
                np.asarray(controller_data["master_right_arm"]["gripper"]).ravel(),
            ])
        }

        return standard_obs

    def reset(self):
        init_qpos = self.robot_config.get("init_qpos", {})
        slave_left_joint: np.array = init_qpos.get("slave_left_arm")[:6]
        slave_right_joint: np.array = init_qpos.get("slave_right_arm")[:6]
        slave_left_gripper: float = init_qpos.get("slave_left_arm")[6]
        slave_right_gripper: float = init_qpos.get("slave_right_arm")[6]

        action = {
            "slave_left_arm": {
                "joint": slave_left_joint,
                "gripper": slave_left_gripper,
            },
            "slave_right_arm": {
                "joint": slave_right_joint,
                "gripper": slave_right_gripper,
            }
        }
        self.move(action)

    def visualize(self):
        data = self.sensor_data
        for cam_name, cam_data in data.items():
            img_bytes = cam_data.get("color")
            if img_bytes is None:
                debug_print(self.name, f"No color image data for {cam_name}. Skipping visualization.", "WARNING")
                continue
            rr.log(f"cameras/{cam_name}", rr.EncodedImage(contents=img_bytes, media_type="image/jpeg"))

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
            
        slave_left = self.controllers["slave_left_arm"]
        slave_right = self.controllers["slave_right_arm"]

        slave_left.set_joint(joint_left)
        slave_right.set_joint(joint_right)

        slave_left.set_gripper(float(gripper_left))
        slave_right.set_gripper(float(gripper_right))

        return