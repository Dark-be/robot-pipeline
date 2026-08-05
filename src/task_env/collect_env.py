import time
import numpy as np
from robot import get_robot
from collector import get_collector
from utils.base.data_handler import debug_print, read_key, KEY_DICT
import rerun as rr
from utils.signals import (
    SIG_EPISODE_START,
    SIG_EPISODE_STOP,
    SIG_QUIT,
    SIG_RESET,
)


# CLI 终端按键 → 采集信号 映射
_KEY_TO_SIGNAL = {
    KEY_DICT["START"]: SIG_EPISODE_START,
    KEY_DICT["STOP"]: SIG_EPISODE_STOP,
    KEY_DICT["QUIT"]: SIG_QUIT,
    KEY_DICT["RESET"]: SIG_RESET,
}


def _cli_signal_source():
    """CLI 模式：把终端按键转换为采集信号。"""
    ch = read_key()
    return _KEY_TO_SIGNAL.get(ch)


class CollectEnv:
    """数据收集环境 —— 组合 Robot + Collector，控制采集流程。

    Robot 只负责管理自身状态（控制器/传感器读写、同步），
    Collector 只负责将数据序列化为 HDF5，
    Env 负责从 Robot 的原始数据中提取出 qpos / action / images。

    signal_source: 无参可调用，非阻塞返回一个采集信号名或 None（默认 CLI 按键转信号）。
                  Web 侧可注入 ControlBus.poll() 实现浏览器信号驱动采集。
    on_frame:   每帧采集回调 on_frame(frame_idx)，用于实时状态推送（Web 侧注入）。
    """

    def __init__(self, base_cfg, signal_source=None, on_frame=None):
        self.name = "CollectEnv"
        self.base_cfg = base_cfg
        self.robot = get_robot(base_cfg)
        self.collector = get_collector(base_cfg)

        self.signal_source = signal_source if signal_source is not None else _cli_signal_source
        self.on_frame = on_frame

        self.collect_env_cfg = self.base_cfg.get("collect_env", {})
        self.save_freq = self.collect_env_cfg.get("save_freq", 30)  # 默认 30 Hz
        if self.save_freq <= 0:
            debug_print(self.name, f"Invalid save_freq={self.save_freq}, reset to 30.", "WARNING")
            self.save_freq = 30

        # Rerun 可视化：每隔 visualize_freq 帧调用 robot.visualize()；
        # 单回合超过 max_frames 帧强制停止；由 web 层置 enable_rerun=True。
        self.visualize_freq = self.collect_env_cfg.get("visualize_freq", 5)
        self.max_frames = self.collect_env_cfg.get("max_frames", 1800)
        self.enable_rerun = False

        self.episode_idx = 0
        self.finish_flag = False
        
        self.visualize_num = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def env_setup(self):
        # 在 env 的 setup 里启动 Rerun 服务（独立子进程 + gRPC 客户端连接）
        if self.enable_rerun:
            try:
                from utils.rerun_launcher import start_rerun
                start_rerun()
            except Exception as exc:  # noqa: BLE001
                debug_print(self.name, f"Rerun start failed: {exc}", "WARNING")
        self.robot.connect()

    def env_finish(self):
        self.robot.disconnect()

    def set_episode_idx(self, idx: int):
        self.episode_idx = idx
    # ------------------------------------------------------------------
    # 采集主循环
    # ------------------------------------------------------------------
    def collect_one_episode(self):
        if self.finish_flag:
            debug_print(self.name, "Data collection has been finished by user. Skipping.", "INFO")
            return

        # --- 复位 + 等待就绪 ---
        self.robot.reset()
        debug_print(self.name, "Waiting for robot ready...", "INFO")
        while not self.robot.is_ready():
            debug_print(self.name, "Robot not started yet, verify hardware.", "WARNING")
            time.sleep(1)

        debug_print(self.name, "Robot READY. episode_start / quit / reset.", "INFO")
        while True:
            sig = self.signal_source()
            if sig == SIG_EPISODE_START:
                break
            if sig == SIG_QUIT:
                debug_print(self.name, "User quit before recording.", "WARNING")
                self.finish_flag = True
                return
            if sig == SIG_RESET:
                debug_print(self.name, "Reset robot.", "INFO")
                self.robot.reset()
            time.sleep(1 / 20)

        # --- 采集循环 ---
        debug_print(self.name, "Recording... Press e to finish.", "INFO")
        collect_num = 0
        while True:
            loop_start = time.monotonic()

            standard_obs = self.robot.get_standard_obs()
            
            self.robot.sync()
            self.collector.collect(standard_obs)

            # 每隔 visualize_freq 帧显示一次 Rerun
            if self.enable_rerun and collect_num % self.visualize_freq == 0:
                rr.set_time("frame", sequence=self.visualize_num)
                self.robot.visualize()
                self.visualize_num += 1

            sig = self.signal_source()
            if sig == SIG_EPISODE_STOP:
                self.collector.finish(self.episode_idx)
                debug_print(self.name, f"Episode {self.episode_idx} finished.", "INFO")
                break
            elif sig == SIG_QUIT:
                debug_print(self.name, "User quit during recording.", "WARNING")
                self.finish_flag = True
                break
            elif sig == SIG_RESET:
                debug_print(self.name, "Reset robot during recording.", "INFO")
                self.robot.reset()
            if self.on_frame is not None:
                self.on_frame(collect_num)
            collect_num += 1

            # 单回合超过 max_frames 帧强制停止（保护数据量与显示）
            if self.max_frames and collect_num >= self.max_frames:
                debug_print(
                    self.name,
                    f"Reached max frames ({self.max_frames}), force stop episode {self.episode_idx}.",
                    "WARNING",
                )
                self.collector.finish(self.episode_idx)
                break

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
