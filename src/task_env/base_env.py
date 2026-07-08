from robot.robot import get_robot
# 基本环境类，定义了环境的基本结构和接口
class BaseEnv:
    def __init__(self, base_cfg):
        self.episode_idx = None
        self.episode_step = 0
        self.base_cfg = base_cfg
        self.finish_flag = False
        self.robot = get_robot(base_cfg=base_cfg)
    
    # 设置环境，初始化机器人和其他必要的组件
    def set_up(self):
        self.robot.set_up()

    # 结束环境，释放资源
    def env_finish(self):
        self.robot.disconnect()
    
    def set_episode_idx(self, idx):
        self.episode_idx = idx
    
    def take_action(self, action):
        self.robot.move(action)
    
    def get_obs(self):
        return self.robot.get_obs()