from .dual_test_robot import Dual_Test_Robot
from .alicia_piper_teleop_robot import Alicia_Piper_Teleop_Robot

ROBOT_REGISTRY = {
    "dual_test_robot": Dual_Test_Robot,
    "alicia_piper_robot": Alicia_Piper_Teleop_Robot,
}

def get_robot(base_cfg):
    robot_config = base_cfg.get("robot")
    robot_type = robot_config.get("type")
    
    if robot_type not in ROBOT_REGISTRY:
        available = list(ROBOT_REGISTRY.keys())
        raise ValueError(f"Can't find robot type '{robot_type}'. Available types are: {available}")
        
    robot_cls = ROBOT_REGISTRY[robot_type]
    
    return robot_cls(robot_config=robot_config)