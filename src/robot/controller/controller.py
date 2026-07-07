import numpy as np
import time
from typing import List
from robot.utils.base.data_handler import debug_print

# 控制器 基类
class Controller:
    # __init__：timestamp 是否给收集消息添加时间戳
    def __init__(self, timestamp=True):
        self.name = "controller"
        self.controller_type = "base_controller"
        # self.is_set_up = False
        self.timestamp = timestamp
    
    # collect_info 需要收集的信息列表
    # eg: ["joint_position", "joint_velocity", "end_effector_position", "end_effector_velocity"]
    def set_collect_info(self, collect_info:List[str]):
        self.collect_info = collect_info
        if self.timestamp:
           self.collect_info.append("timestamp")
    
    # get_information 获取控制器信息的函数 需要子类实现
    def get_information(self):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    # get 获取控制器信息的接口函数，调用get_information获取信息，并根据collect_info返回需要的Dict
    def get(self):
        if self.collect_info is None:
            raise ValueError(f"{self.name}: collect_info is not set")
        info = self.get_information()

        if self.timestamp:
            info["timestamp"] = time.monotonic_ns()
        # 检查collect_info键中的信息是否为None，如果是则打印错误日志
        for collect_info in self.collect_info:
            if info[collect_info] is None:
                debug_print(f"{self.name}", f"{collect_info} information is None", "ERROR")
        
        debug_print(f"{self.name}", f"get data: {info} ", "DEBUG")
        return {collect_info: info[collect_info] for collect_info in self.collect_info}
    
    # move_controller 控制器执行动作的函数 需要子类实现
    # move_data: Dict[str, Any] 包含控制器需要执行的动作信息的字典
    # is_delta: bool 表示move_data中的动作信息是否为增量
    def move_controller(self, move_data, is_delta=False):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def move(self, move_data, is_delta=False):
        debug_print(f"{self.name}", f"get move data: {move_data} ", "DEBUG")
        try:
            self.move_controller(move_data, is_delta)
        except Exception as e:
            debug_print(self.name, f"move error: {e}", "WARNING")
    
   # init controller
    def set_up(self):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def disconnect(self):
        debug_print(self.name, "Controller has no resources to disconnect", "WARNING")
        pass
    # print controller
    # __repr__：打印控制器信息
    def __repr__(self):
        return f"Base Controller, can't be used directly \n \
                name: {self.name} \n \
                controller_type: {self.controller_type}"