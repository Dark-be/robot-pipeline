import time
import numpy as np
from robot import get_robot
from collector import get_collector
from utils.base.data_handler import debug_print, read_key, KEY_DICT
import rerun as rr

class CollectEnv:
    """数据收集环境 —— 组合 Robot + Collector，控制采集流程。

    Robot 只负责管理自身状态（控制器/传感器读写、同步），
    Collector 只负责将数据序列化为 HDF5，
    Env 负责从 Robot 的原始数据中提取出 qpos / action / images。
    """

    def __init__(self, base_cfg):
        self.name = "CollectEnv"
        self.base_cfg = base_cfg
        self.robot = get_robot(base_cfg)
        self.collector = get_collector(base_cfg)

        self.collect_env_cfg = self.base_cfg.get("collect_env", {})
        if not self.collect_env_cfg:
            raise ValueError("collect_env config is required in base_cfg.")
        
        self.save_freq = self.collect_env_cfg.get("save_freq", 30)  # 默认 30 Hz
        if self.save_freq <= 0:
            debug_print(self.name, f"Invalid save_freq={self.save_freq}, reset to 30.", "WARNING")
            self.save_freq = 30

        self.enable_rerun = self.collect_env_cfg.get("enable_rerun", False)

        self.episode_idx = 0
        self.finish_flag = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def env_setup(self):
        rr.init("collect_env", spawn=False)
        server_url = rr.serve_grpc(
            grpc_port=9876,
            server_memory_limit="1GiB",
            newest_first=False,
            cors_allow_origin=["*"]
        )
        rr.serve_web_viewer(
            web_port=9090,  # Web 界面端口
            open_browser=False,  # 自动打开浏览器
            connect_to=server_url  # 连接到 gRPC 服务器
        )
        debug_print(self.name, f"Visualization already started: {server_url}", "INFO")
        self.robot.connect()

    def env_finish(self):
        self.robot.disconnect()
        rr.disconnect()

    def set_episode_idx(self, idx: int):
        self.episode_idx = idx
    # ------------------------------------------------------------------
    # 采集主循环
    # ------------------------------------------------------------------
    def collect_one_episode(self):
        """采集一个 episode 的数据。
        流程:
          1. 复位机器人
          2. 等待就绪 + 按 s 开始
          3. 循环: get_obs → sync → 提取 qpos/action/images → collector.collect
          4. 按 e 结束 → collector.finish()
        """
        if self.finish_flag:
            debug_print(self.name, "Data collection has been finished by user. Skipping.", "INFO")
            return

        # --- 复位 + 等待就绪 ---
        self.robot.reset()
        debug_print(self.name, "Waiting for robot ready...", "INFO")
        while not self.robot.is_ready():
            debug_print(self.name, "Robot not started yet, verify hardware.", "WARNING")
            time.sleep(1)

        debug_print(self.name, "Robot READY. Press s to start, q to quit.", "INFO")
        while True:
            ch = read_key()
            if ch == KEY_DICT["START"]:
                break
            if ch == KEY_DICT["QUIT"]:
                debug_print(self.name, "User quit before recording.", "WARNING")
                self.finish_flag = True
                return
            time.sleep(1 / 20)

        # --- 采集循环 ---
        debug_print(self.name, "Recording... Press e to finish.", "INFO")
        collect_num = 0
        while True:
            loop_start = time.monotonic()

            standard_obs = self.robot.get_standard_obs()
            self.robot.sync()
            self.collector.collect(standard_obs)
            if self.enable_rerun and collect_num % 15 == 0:
                rr.set_time("frame", sequence=collect_num)
                self.robot.visualize()

            ch = read_key()
            if ch == KEY_DICT["END"]:
                self.collector.finish(self.episode_idx)
                debug_print(self.name, f"Episode {self.episode_idx} finished.", "INFO")
                break
            if ch == KEY_DICT["VISUALIZE"]:
                self.enable_rerun = not self.enable_rerun
                debug_print(self.name, f"Rerun visualization {'enabled' if self.enable_rerun else 'disabled'}.", "INFO")
            elif ch == KEY_DICT["QUIT"]:
                debug_print(self.name, "User quit during recording.", "WARNING")
                self.finish_flag = True
                break

            collect_num += 1

            elapsed = time.monotonic() - loop_start
            wait = 1.0 / self.save_freq - elapsed
            if wait > 0:
                time.sleep(wait)
            else:
                debug_print(self.name, f"Collect over limit: {elapsed:.3f}s > {1/self.save_freq:.3f}s", "WARNING")

        if collect_num > 0:
            debug_print(self.name, f"Episode done: {collect_num} frames", "INFO")
        else:
            debug_print(self.name, "No frames collected this episode.", "WARNING")
