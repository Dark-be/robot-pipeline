# sensor 基类，所有 sensor 都继承自该类
# connect 和 disconnect 方法需要在子类中实现，用于连接和断开传感器
# 每个 sensor 都有一个唯一的 name
# 接受 collect_info 参数，表示需要收集哪些信息
# get_information 方法返回一个字典，根据 collect_info 的内容返回对应的信息
# get 方法根据 collect_info 调用 get_information，并返回一个字典，包含需要收集的信息
from utils.base.data_handler import debug_print

class Sensor:
    def __init__(self, name="sensor"):
        self.name = name
        self.collect_info = None
    
    def set_collect_info(self, collect_info: list[str]):
       self.collect_info = collect_info

    def get(self):
        if self.collect_info is None:
            debug_print(self.name, f"collect_info is not set", "WARNING")
            return None
        
        info = self.get_information()
        if info is None:
            debug_print(self.name, f"get_information returned None", "ERROR")
            return None

        return info


    def connect(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def disconnect(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def get_information(self):
        raise NotImplementedError("This method should be implemented by the subclass")