from .base_robot import *

#from .dual_x_arm import Dual_X_Arm
from .dual_test_robot import Dual_Test_Robot
from .dual_alicia_teleop_robot import Dual_Alicia_Teleop_Robot
from .alicia_piper_robot import Alicia_Piper_Robot

ROBOT_REGISTRY = {
    "alicia_piper_robot": Alicia_Piper_Robot,
    "dual_test_robot": Dual_Test_Robot,
    "dual_alicia_teleop_robot": Dual_Alicia_Teleop_Robot,
}


def get_robot(base_cfg):
    robot_type = base_cfg["robot"].get("type")
    
    # 1. 检查配置是否存在
    if not robot_type:
        raise KeyError("配置文件中缺少 ['robot']['type'] 字段，请检查您的 config.yml")
        
    # 2. 检查注册表
    if robot_type not in ROBOT_REGISTRY:
        available = list(ROBOT_REGISTRY.keys())
        raise ValueError(f"未找到机器人类型 '{robot_type}'。当前已注册的可选类型有: {available}")
        
    robot_cls = ROBOT_REGISTRY[robot_type]
    
    return robot_cls(base_config=base_cfg)