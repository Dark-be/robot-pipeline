from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

from robot.robot.base_robot import Robot
from robot.controller.Alicia_teach_controller import AliciaTeachController
from robot.controller.Piper_controller import PiperController

from robot.sensor.v4l2_sensor import V4l2Sensor
from robot.sensor.realsense_sensor import RealsenseSensor

from robot.utils.base.data_handler import debug_print
from robot.utils.kenimatics import get_tool_position

import rerun as rr

class Dual_Alicia_Teleop_Robot(Robot):

    def __init__(self, base_config):
        super().__init__(base_config=base_config)

        self.controllers = {
            "arm" : {
                "master_left_arm": AliciaTeachController("master_left_arm"),
                "master_right_arm": AliciaTeachController("master_right_arm"),
                "slave_left_arm": PiperController("slave_left_arm", INFO="DEBUG"),
                "slave_right_arm": PiperController("slave_right_arm", INFO="DEBUG"),
            },
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_left_wrist": V4l2Sensor("cam_left_wrist"),
                "cam_right_wrist": V4l2Sensor("cam_right_wrist"),
            },
        }

    def set_up(self):
        super().set_up()

        master_port_left = self.robot_config["MASTER_PORT"]["left_arm"]
        master_port_right = self.robot_config["MASTER_PORT"]["right_arm"]
        slave_port_left = self.robot_config["SLAVE_PORT"]["left_arm"]
        slave_port_right = self.robot_config["SLAVE_PORT"]["right_arm"]
        if not master_port_left:
            debug_print(self.type, "MASTER_PORT for left_arm is not configured. Please check your config.yml", "ERROR")
        if not master_port_right:
            debug_print(self.type, "MASTER_PORT for right_arm is not configured. Please check your config.yml", "ERROR")
        if not slave_port_left:
            debug_print(self.type, "SLAVE_PORT for left_arm is not configured. Please check your config.yml", "ERROR")
        if not slave_port_right:
            debug_print(self.type, "SLAVE_PORT for right_arm is not configured. Please check your config.yml", "ERROR")

        head_cam_serial = self.robot_config["CAMERA_SERIALS"]["head"]
        left_cam_serial = self.robot_config["CAMERA_SERIALS"]["left"]
        right_cam_serial = self.robot_config["CAMERA_SERIALS"]["right"]
        if not head_cam_serial:
            debug_print(self.type, "CAMERA_SERIALS for head is not configured. Please check your config.yml", "ERROR")
        if not left_cam_serial:
            debug_print(self.type, "CAMERA_SERIALS for left is not configured. Please check your config.yml", "ERROR")
        if not right_cam_serial:
            debug_print(self.type, "CAMERA_SERIALS for right is not configured. Please check your config.yml", "ERROR")

        self.controllers["arm"]["master_left_arm"].set_up(port=master_port_left)
        self.controllers["arm"]["master_right_arm"].set_up(port=master_port_right)
        self.controllers["arm"]["slave_left_arm"].set_up(port=slave_port_left)
        self.controllers["arm"]["slave_right_arm"].set_up(port=slave_port_right)

        self.sensors["image"]["cam_head"].set_up(device=head_cam_serial, is_depth=False, is_jpeg=True)
        self.sensors["image"]["cam_left_wrist"].set_up(device=left_cam_serial, is_depth=False, is_jpeg=True)
        self.sensors["image"]["cam_right_wrist"].set_up(device=right_cam_serial, is_depth=False, is_jpeg=True)

        # 设置数据记录类型，slave 同时记录关节和夹爪状态
        self.set_collect_type({"arm": ["joint", "gripper", "eef"], "image": ["color"]})
        debug_print(self.type, f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Setup complete.", "INFO")
        time.sleep(1)

    def reset(self):
        super().reset()

        init_qpos = self.robot_config.get("init_qpos", {})
        slave_left_joint: List[float] = init_qpos.get("slave_left_arm")
        slave_right_joint: List[float] = init_qpos.get("slave_right_arm")
        slave_left_gripper: float = init_qpos.get("slave_left_gripper")
        slave_right_gripper: float = init_qpos.get("slave_right_gripper")

        move_data = {
            "arm": {
                "slave_left_arm": {
                    "joint": slave_left_joint,
                    "gripper": slave_left_gripper,
                },
                "slave_right_arm": {
                    "joint": slave_right_joint,
                    "gripper": slave_right_gripper,
                }
            }
        }
        self.move(move_data)

    def visualize(self):
        data = self.sensor_data
        for cam_name, cam_data in data.items():
            img_bytes = cam_data.get("color")
            if img_bytes is None:
                debug_print(self.type, f"No color image data for {cam_name}. Skipping visualization.", "WARNING")
                continue
            rr.log(f"cameras/{cam_name}", rr.EncodedImage(contents=img_bytes, media_type="image/jpeg"))

    def sync(self):
        # start_time = time.monotonic()
        master_left_data = self.controller_data.get("master_left_arm", {})
        master_right_data = self.controller_data.get("master_right_arm", {})
        if not master_left_data or not master_right_data:
            debug_print(self.type, "No data received from master controllers. Skipping sync.", "WARNING")
            return
        
        joint_left = master_left_data.get("joint")
        joint_right = master_right_data.get("joint")
        if joint_left is None or joint_right is None:
            debug_print(self.type, "Joint data from master controllers is missing. Skipping sync.", "WARNING")
            return
        
        

        gripper_left = master_left_data.get("gripper")
        gripper_right = master_right_data.get("gripper")
        if gripper_left is None or gripper_right is None:
            debug_print(self.type, "Gripper data from master controllers is missing. Skipping sync.", "WARNING")
            return
            
        slave_left = self.controllers["arm"]["slave_left_arm"]
        slave_right = self.controllers["arm"]["slave_right_arm"]
        
        slave_left.set_joint(joint_left)
        slave_right.set_joint(joint_right)

        slave_left.set_gripper(float(gripper_left))
        slave_right.set_gripper(float(gripper_right))
        # get_time = time.monotonic() - start_time

        # elapsed_time = time.monotonic() - start_time
        # debug_print(self.type, f"Sync time: {elapsed_time:.4f} seconds", "WARNING")
        # debug_print(self.type, f"Get master data time: {get_time:.4f} seconds", "WARNING")
        return