import numpy as np
from controller.test_arm_controller import TestArmController
from sensor.test_vision_sensor import TestVisonSensor
from robot.base_robot import Robot
from utils.base.data_handler import debug_print


class Dual_Test_Robot(Robot):
    def __init__(self, robot_config: dict):
        super().__init__(robot_config=robot_config)
        
        self.controllers = {
            "left_arm": TestArmController("left_arm"),
            "right_arm": TestArmController("right_arm"),
        }
        self.sensors = {
            "cam_head": TestVisonSensor("cam_head"),
            "cam_left_wrist": TestVisonSensor("cam_left_wrist"),
            "cam_right_wrist": TestVisonSensor("cam_right_wrist"),
        }


    def get_standard_obs(self):
        obs = self.get_obs()
        controller_data, sensor_data = obs
        standard_obs = {
            "observations/qpos": np.concatenate([
                np.asarray(controller_data["left_arm"]["joint"]).ravel(),
                np.asarray(controller_data["left_arm"]["gripper"]).ravel(),
                np.asarray(controller_data["right_arm"]["joint"]).ravel(),
                np.asarray(controller_data["right_arm"]["gripper"]).ravel(),
            ]),
            "observations/images/cam_head": sensor_data["cam_head"]["color"],
            "observations/images/cam_left_wrist": sensor_data["cam_left_wrist"]["color"],
            "observations/images/cam_right_wrist": sensor_data["cam_right_wrist"]["color"],
            "action": np.concatenate([
                np.asarray(controller_data["left_arm"]["joint"]).ravel(),
                np.asarray(controller_data["left_arm"]["gripper"]).ravel(),
                np.asarray(controller_data["right_arm"]["joint"]).ravel(),
                np.asarray(controller_data["right_arm"]["gripper"]).ravel(),
            ])
        }
        return standard_obs

    def connect(self):
        self.controllers["left_arm"].connect()
        self.controllers["left_arm"].set_collect_info(["joint", "gripper", "pose"])

        self.controllers["right_arm"].connect()
        self.controllers["right_arm"].set_collect_info(["joint", "gripper", "pose"])

        self.sensors["cam_head"].connect(is_jpeg=True)
        self.sensors["cam_head"].set_collect_info(["color"])

        self.sensors["cam_left_wrist"].connect(is_jpeg=True)
        self.sensors["cam_left_wrist"].set_collect_info(["color"])

        self.sensors["cam_right_wrist"].connect(is_jpeg=True)
        self.sensors["cam_right_wrist"].set_collect_info(["color"])

        debug_print(self.name, f"Setup complete.", "INFO")
        self.ready = True

    def reset(self):
        action = {
            "left_arm":{
                "joint": self.robot_config['init_qpos']['left_arm'][:6],
                "gripper":  self.robot_config['init_qpos']['left_arm'][6],
            },
            "right_arm":{
                "joint": self.robot_config['init_qpos']['right_arm'][:6],
                "gripper": self.robot_config['init_qpos']['right_arm'][6],
            }
        }
        self.take_action(action)