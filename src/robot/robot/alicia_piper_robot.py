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


class Alicia_Piper_Robot(Robot):

    def __init__(self, base_config):
        super().__init__(base_config=base_config)

        self.controllers = {
            "arm" : {
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

        slave_port_left = self.robot_config["SLAVE_PORT"]["left_arm"]
        slave_port_right = self.robot_config["SLAVE_PORT"]["right_arm"]

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

        self.controllers["arm"]["slave_left_arm"].set_up(port=slave_port_left)
        self.controllers["arm"]["slave_right_arm"].set_up(port=slave_port_right)

        self.sensors["image"]["cam_head"].set_up(device=head_cam_serial, is_depth=False, is_jpeg=False)
        self.sensors["image"]["cam_left_wrist"].set_up(device=left_cam_serial, is_depth=False, is_jpeg=False)
        self.sensors["image"]["cam_right_wrist"].set_up(device=right_cam_serial, is_depth=False, is_jpeg=False)

        # 设置数据记录类型，slave 同时记录关节和夹爪状态
        self.set_collect_type({"arm": ["joint", "gripper"], "image": ["color"]})
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