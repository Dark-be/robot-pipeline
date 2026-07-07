import time
from robot.utils.base.data_handler import debug_print, read_key, KEY_DICT
from .base_env import BaseEnv
# 数据收集环境，负责控制数据收集的流程和逻辑
class CollectEnv(BaseEnv):
    def __init__(self, base_cfg):
        super().__init__(base_cfg=base_cfg)
        self.success_num, self.episode_num = 0, 0
        self.finish_flag = False

    # 收集一个回合的数据，直到用户按下Enter键或脚踏开关触发结束
    def collect_one_episode(self):
        if self.finish_flag:
            debug_print("COLLECT", "Data collection has been finished by user. Skipping collect_one_episode.", "INFO")
            return
        # 重置机器人状态，等待用户准备好并按下Enter键开始数据收集
        self.robot.reset()
        debug_print("COLLECT", "Waiting for robot ready and Enter key...", "INFO")
        
        # 检查配置文件中是否指定了数据收集的频率，默认为30Hz
        if 'collect' not in self.base_cfg or 'save_freq' not in self.base_cfg['collect']:
            debug_print("COLLECT", "Missing 'save_freq' in config. Using default 30Hz.", "WARNING")
            save_freq = 30
        else:
            save_freq = self.base_cfg['collect']["save_freq"]
            if save_freq <= 0:
                debug_print("COLLECT", f"Invalid save_freq: {save_freq}. Resetting to 30Hz.", "ERROR")
                save_freq = 30
        # 等待机器人准备好，如果使用脚踏开关则等待脚踏开关触发，否则等待用户按下Enter键
        while not self.robot.is_start():
            debug_print("COLLECT", "Robot not started yet, verify hardware connection.", "WARNING")
            time.sleep(1)
        debug_print("COLLECT", "Robot READY. Press s to start recording... or q to quit.", "INFO")
        run_flag = True
        while run_flag:
            ch = read_key()
            if ch == KEY_DICT["START"]:
                run_flag = False
            if ch == KEY_DICT["QUIT"]:
                debug_print("COLLECT", "Data collection interrupted by user (q pressed). Exiting.", "WARNING")
                self.finish_flag = True
                return
            time.sleep(1 / 20)
        
        debug_print("COLLECT", "Recording... Press e to finish.", "INFO")

        avg_collect_time, collect_num = 0.0, 0
        while True:
            start_time = time.monotonic()
            data = self.robot.get_obs()

            self.robot.sync()
            self.robot.collect(data)
            # 检查用户是否按下结束数据收集的键
            ch = read_key()
            if ch == KEY_DICT["END"]:
                self.robot.collect_finish(self.episode_idx)
                break
            collect_num += 1

            # 控制数据收集的频率，确保按照指定的save_freq进行数据保存
            eplased_time = time.monotonic() - start_time
            wait_time = 1 / save_freq - eplased_time
            avg_collect_time += eplased_time
            if wait_time <= 0:
                debug_print("COLLECT", f"Collecting time over limit. t={eplased_time}", "WARNING")
            else:
                debug_print("COLLECT", f"Wait_time: {wait_time} / {1 / save_freq}", "INFO")
                time.sleep(wait_time)
            # debug_print("COLLECT", f"get obs time: {get_obs_time}", "WARNING")
            # debug_print("COLLECT", f"sync time: {sync_time}", "WARNING")
            # debug_print("COLLECT", f"frame time: {current_time-last_time}", "WARNING")

        # 额外信息记录：
        # 1. 本回合数据收集的总次数（collect_num），即在本回合中调用了多少次self.robot.collect(data)
        # 2. 平均数据收集时间间隔（avg_time_interval），即每次数据收集的平均时间，单位为秒
        extra_info = {}
        if collect_num == 0:
            debug_print("COLLECT", "No data collected during this episode. Setting avg_time_interval to 0.", "WARNING")
            avg_collect_time = 0.0
        else:
            avg_collect_time = avg_collect_time / collect_num
            extra_info["collect_num"] = collect_num
            extra_info["avg_time_interval"] = avg_collect_time
        self.robot.collector.add_extra_cfg_info(extra_info)