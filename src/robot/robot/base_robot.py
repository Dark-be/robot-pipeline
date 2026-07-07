from typing import Dict, Any
import time
import numpy as np
from robot.data.collect_any import CollectAny
from robot.utils.base.data_handler import debug_print, hdf5_groups_to_dict, dict_to_list

# add your controller/sensor type here
# 采集的数据种类
ALLOW_TYPES = ["arm", "image"]
KEY_BANNED = ["timestamp"] # 在判断是否停止时忽略这些键的变化，因为它们可能会频繁变化但不代表实际动作
    
class Robot:
    def __init__(self, base_config) -> None:
        if "robot" not in base_config:
            debug_print("ROBOT", "Missing 'robot' section in config!", "ERROR")
            raise KeyError("Config missing 'robot' section")
            
        self.robot_config = base_config["robot"]
        self.type = self.robot_config.get("type", "unknown_robot")
        
        self.controllers = {}
        self.sensors = {}

        # 如果配置文件中存在"collect"部分，则初始化数据收集器CollectAny，否则设置为None并在相关方法中抛出错误提示
        if "collect" in base_config:
            debug_print(self.type, "Found 'collect' section in config.", "INFO")
            self.collect_cfg = base_config["collect"]
            debug_print(self.type, f"Collect_cfg: \n {self.collect_cfg}", "INFO")
            self.collector = CollectAny(self.collect_cfg)

        if not self.collect_cfg:
            debug_print(self.type, "Can't find any valid section in config, Please check your config file.", "ERROR")
        
        self.move_tolerance = self.robot_config.get("move_tolerance", 0.01)
        self.bias = self.robot_config.get("bias", None)

        self.last_controller_data = {}
        self.controller_data = {}
        self.sensor_data = {}

    # 检查机器人初始化部件中是否有不符合规范的控制器或传感器类型，并打印警告信息
    def set_up(self):
        for controller_type in self.controllers.keys():
            if controller_type not in ALLOW_TYPES:
                debug_print(self.type, f"It's recommanded to set your controller type into our format.\n\
                            YOUR STPE:{controller_type}\n\
                            ALLOW_TYPES:{ALLOW_TYPES}", "WARNING")
        
        for sensor_type in self.sensors.keys():
            if sensor_type not in ALLOW_TYPES:
                debug_print(self.type, f"It's recommanded to set your sensor type into our format.\n\
                            YOUR STPE:{sensor_type}\n\
                            ALLOW_TYPES:{ALLOW_TYPES}", "WARNING")
    
    # 采集部分
    # 设置各个部件的采集信息类型，INFO_NAMES是一个字典，键为部件类型，值为对应的采集信息
    def set_collect_type(self, INFO_NAMES: Dict[str, Any]):
        for key, value in INFO_NAMES.items():
            if key in self.controllers:
                for controller in self.controllers[key].values():
                    controller.set_collect_info(value)
            if key in self.sensors:
                for sensor in self.sensors[key].values():
                    sensor.set_collect_info(value)
    
    # 调用 collector 采集一次数据
    def collect(self, data):
        if self.collector is None:
            raise ValueError("Can't find collector!")
        self.collector.collect(data[0], data[1])

    # 完成一次数据采集，写入额外文件
    # 包括各个部件的采集信息类型
    def collect_finish(self, episode_id=None):
        if self.collector is None:
            raise ValueError("Can't find collector!")
        
        extra_info = {}
        for controller_type in self.controllers.keys():
            extra_info[controller_type] = []
            for key in self.controllers[controller_type].keys():
                extra_info[controller_type].append(key)

        for sensor_type in self.sensors.keys():
            extra_info[sensor_type] = []
            for key in self.sensors[sensor_type].keys():
                extra_info[sensor_type].append(key)

        self.collector.add_extra_cfg_info(extra_info, repeat=False)
        # 写入 episode.hdf5
        self.collector.write(episode_id)

    # 目前用于遥操作时的主从臂同步
    def sync(self):
        return
    
    # 通用部分
    # 返回当前机器人的观测数据，包括控制器数据和传感器数据。如果任何传感器返回None，则返回[None, None]，表示数据获取失败。
    def get_obs(self):
        if self.controllers is not None:
            for type_name, controller_type in self.controllers.items():
                for controller_name, controller in controller_type.items():
                    self.controller_data[controller_name] = controller.get()
        if self.sensors is not None:
            for type_name, sensor_type in self.sensors.items(): 
                for sensor_name, sensor in sensor_type.items():
                    self.sensor_data[sensor_name] = sensor.get()
        
        return [self.controller_data.copy(), self.sensor_data.copy()]
    
    # 主动断开所有控制器的连接，释放资源。遍历所有控制器类型和控制器实例，调用每个控制器的disconnect方法，并打印调试信息。
    def disconnect(self):
        for controller_type in self.controllers.values():
            for controller in controller_type.values():
                controller.disconnect()

        debug_print(self.type, "All controllers have been disconnected.", "INFO")
    
    def move(self, move_data, key_banned=None):
        if move_data is None:
            return
        
        for controller_type_name, controller_type in move_data.items():
            for controller_name, controller_action in controller_type.items():
                if self.bias:
                    if controller_name in self.bias.keys():
                        for k in self.bias[controller_name].keys():
                            if k in controller_action.keys():
                                controller_action[k] += self.bias[controller_name][k]
                if key_banned is None:        
                    self.controllers[controller_type_name][controller_name].move(controller_action, is_delta=False)
                else:
                    controller_action = remove_duplicate_keys(controller_action, key_banned)
                    self.controllers[controller_type_name][controller_name].move(controller_action, is_delta=False)

    # 检查 10 次，如果连续 10 次都没有检测到移动，则认为移动完成，退出循环
    def move_blocking(self, move_data, check_freq=30, key_banned=None):
        stop_num = 0
        self.move(move_data, key_banned=key_banned)

        while True:
            time.sleep(1 / check_freq)

            if not self.is_move():
                stop_num += 1
            else:
                stop_num = 0
            
            if stop_num > 10:
                break

    def is_start(self):
        debug_print(self.type, "your are using is_start(), this will return True.", "DEBUG")
        return True

    def reset(self):
        debug_print(self.type, "your are using reset(), this will return True.", "DEBUG")
        return True
    
    def is_move(self):
        controller_data = {}
        for type_name, controller_type in self.controllers.items():
            for controller_name, controller in controller_type.items():
                controller_data[controller_name] = controller.get()
        
        if not self.last_controller_data:
            self.last_controller_data = controller_data
            return True
        else:
            for part, current_subdata in controller_data.items():
                previous_subdata = self.last_controller_data.get(part)
                if previous_subdata is None:
                    return True

                if isinstance(current_subdata, dict):
                    for key, current_value in current_subdata.items():
                        if key in KEY_BANNED:
                            continue
                        
                        previous_value = previous_subdata.get(key)
                        if previous_value is None:
                            return True 

                        current_arr = np.atleast_1d(current_value)
                        previous_arr = np.atleast_1d(previous_value)

                        if current_arr.shape != previous_arr.shape:
                            self.last_controller_data = controller_data
                            return True 

                        if np.any(np.abs(current_arr - previous_arr) > self.move_tolerance):
                            self.last_controller_data = controller_data
                            return True 
                else:
                    current_arr = np.atleast_1d(current_subdata)
                    previous_arr = np.atleast_1d(previous_subdata)

                    if current_arr.shape != previous_arr.shape:
                        self.last_controller_data = controller_data
                        return True

                    if np.any(np.abs(current_arr - previous_arr) > self.move_tolerance):
                        self.last_controller_data = controller_data
                        return True
            return False

    # 回放部分
    def replay(self, data_path, fps=30, key_banned=None, is_collect=False, episode_id=None):
        time_interval = 1 / fps
        episode_data = dict_to_list(hdf5_groups_to_dict(data_path))
        
        now_time = time.monotonic()
        for current_action in episode_data:
            start_time = time.monotonic()
            if is_collect:
                data = self.get_obs()
                self.collect(data)
            
            self.play_once(current_action, key_banned)

            elpased_time = now_time - start_time
            while elpased_time < time_interval:
                now_time = time.monotonic()
                time.sleep(time_interval - elpased_time)
        if is_collect:
            self.finish(episode_id)
    
    def play_once(self, episode: Dict[str, Any], key_banned=None):
        for controller_type, controller_group in self.controllers.items():
            for controller_name, controller in controller_group.items():
                if controller_name in episode:
                    controller_action = episode[controller_name]
                    move_data = {
                        controller_type: {
                            controller_name: controller_action,
                        },
                    }
                    # print(f"Playing back action for {controller_name}: {controller_action}")
                    # print(f"key_banned: {key_banned}")
                    # print(f"move_data before removing keys: {move_data}")
                    self.move(move_data, key_banned=key_banned)

def remove_duplicate_keys(source_dict, keys_to_remove):
    return {k: v for k, v in source_dict.items() if k not in keys_to_remove}

