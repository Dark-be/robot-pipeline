# controller 基类，所有 controller 都继承自该类
# connect 和 disconnect 方法需要在子类中实现，用于连接和断开控制器
# 接受 collect_info 参数，表示需要收集哪些信息

# take_action 控制器执行动作的函数 需要子类实现
# action: Dict[str, Any] 包含控制器需要执行的动作信息的字典
# is_delta: bool 表示action中的动作信息是否为增量
from typing import List
from utils.base.data_handler import debug_print

class Controller:
    def __init__(self, name="controller"):
        self.name = name
        self.collect_info = None
    
    def set_collect_info(self, collect_info: List[str]):
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
    
    def take_action(self, action):
        raise NotImplementedError("This method should be implemented by the subclass")
    