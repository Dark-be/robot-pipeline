import time
from robot.utils.base.data_handler import debug_print, read_key, KEY_DICT
from .base_env import BaseEnv
import rerun as rr
from .preview_worker import PreviewWorker

# 数据收集环境，负责控制数据收集的流程和逻辑
class CollectEnv(BaseEnv):
    # def __init__(self, base_cfg):
    #     super().__init__(base_cfg=base_cfg)
    #     self.enable_rerun = base_cfg.get("enable_rerun", False)

    def __init__(self, base_cfg):
        super().__init__(base_cfg=base_cfg)

        collect_cfg = base_cfg.get("collect", {})
        preview_cfg = collect_cfg.get("preview", {})

        self.enable_rerun = bool(
            preview_cfg.get("enabled", True)
        )

        self.preview_fps = float(
            preview_cfg.get("fps", 15.0)
        )

        if self.preview_fps <= 0:
            debug_print(
                "COLLECT",
                (
                    f"Invalid preview fps={self.preview_fps}; "
                    "resetting to 8 Hz."
                ),
                "WARNING",
            )
            self.preview_fps = 8.0

        self.preview_grpc_port = int(
            preview_cfg.get("grpc_port", 9876)
        )
        self.preview_web_port = int(
            preview_cfg.get("web_port", 9090)
        )
        self.preview_open_browser = bool(
            preview_cfg.get("open_browser", True)
        )
        self.preview_memory_limit = str(
            preview_cfg.get("memory_limit", "512MiB")
        )

        self.preview_worker: PreviewWorker | None = None
        self.rerun_recording: rr.RecordingStream | None = None

    
    def set_up(self):
        super().set_up()
        # if self.enable_rerun:
        #     rr.init("collection", spawn=False)
        #     server_url = rr.serve_grpc(
        #         grpc_port=9876,
        #         server_memory_limit="1GiB",
        #         newest_first=False,
        #         cors_allow_origin=["*"]
        #     )
        #     debug_print("COLLECT", f"Rerun gRPC 服务器已启动: {server_url}", "INFO")
        #     rr.serve_web_viewer(
        #         web_port=9090,  # Web 界面端口
        #         open_browser=False,  # 自动打开浏览器
        #         # connect_to=server_url  # 连接到 gRPC 服务器
        #     )

        if self.enable_rerun:
            # 保存明确的 RecordingStream，后面传给 PreviewWorker。
            self.rerun_recording = rr.RecordingStream(
                application_id="collection",
                make_default=True,
            )

            server_url = self.rerun_recording.serve_grpc(
                grpc_port=self.preview_grpc_port,
                server_memory_limit=self.preview_memory_limit,
            )

            debug_print(
                "COLLECT",
                (
                    f"Rerun recording initialized: "
                    f"recording_id="
                    f"{self.rerun_recording.get_recording_id()}"
                ),
                "INFO",
            )

            debug_print(
                "COLLECT",
                f"Rerun gRPC server started: {server_url}",
                "INFO",
            )

            rr.serve_web_viewer(
                web_port=self.preview_web_port,
                open_browser=self.preview_open_browser,
                connect_to=server_url,
            )

            # 先发送一张测试图。
            # 如果它能显示，说明 Viewer 和 gRPC 链路正常。
            import numpy as np

            test_image = np.zeros(
                (120, 160, 3),
                dtype=np.uint8,
            )

            # 加一个白色区域，避免全黑画面难以辨认。
            test_image[20:100, 20:140] = 255

            self.rerun_recording.log(
                "debug/connection_test",
                rr.Image(test_image),
                static=True,
            )

            debug_print(
                "COLLECT",
                "Rerun connection test image logged.",
                "INFO",
            )

            self.preview_worker = PreviewWorker(
                recording=self.rerun_recording,
            )
            self.preview_worker.start()

            debug_print(
                "COLLECT",
                (
                    f"Camera preview enabled: "
                    f"fps={self.preview_fps}, "
                    f"web=http://127.0.0.1:"
                    f"{self.preview_web_port}"
                ),
                "INFO",
            )

    def env_finish(self):
        try:
            if self.preview_worker is not None:
                self.preview_worker.close()
                self.preview_worker = None
        finally:
            super().env_finish()

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
        if save_freq <= 0:
            debug_print(
                "COLLECT",
                f"Invalid save_freq: {save_freq}. Resetting to 30Hz.",
                "ERROR",
            )
            save_freq = 30
        # 预览频率不能超过数据采集频率。
        effective_preview_fps = min(
            float(save_freq),
            self.preview_fps,
        )

        # 使用累加器，而不是简单的 collect_num % N。
        # 例如采集 30Hz、预览 8Hz 时，可以更准确地保持平均 8Hz。
        preview_accumulator = (
            float(save_freq) - effective_preview_fps
        )

        if self.enable_rerun:
            debug_print(
                "COLLECT",
                (
                    f"Collection FPS={save_freq}, "
                    f"preview FPS={effective_preview_fps}"
                ),
                "INFO",
            )
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

        # while True:
        #     rr.set_time("frame", sequence=collect_num)
        #     start_time = time.monotonic()
        #     data = self.robot.get_obs()

        #     self.robot.sync()
        #     self.robot.collect(data)

        #     if self.enable_rerun:
        #         self.robot.visualize()
        while True:
            start_time = time.monotonic()

            data = self.robot.get_obs()
            self.robot.sync()
            self.robot.collect(data)

            # 只控制预览提交频率，不影响实际数据采集频率。
            if (
                self.enable_rerun
                and self.preview_worker is not None
            ):
                preview_accumulator += effective_preview_fps

                if preview_accumulator >= float(save_freq):
                    self.preview_worker.submit(
                        frame_index=collect_num,
                        sensor_data=self.robot.sensor_data,
                    )
                    preview_accumulator -= float(save_freq)

            # 检查用户是否按下结束数据收集的键
            ch = read_key()
            if ch == KEY_DICT["END"]:
                self.robot.collect_finish(self.episode_idx)
                break
            elif ch == KEY_DICT["QUIT"]:
                debug_print("COLLECT", "Data collection interrupted by user (q pressed). Exiting.", "WARNING")
                self.finish_flag = True
                return
            
            collect_num += 1

            # 控制数据收集的频率，确保按照指定的save_freq进行数据保存
            eplased_time = time.monotonic() - start_time
            wait_time = 1 / save_freq - eplased_time
            # 超时
            if wait_time <= 0:
                avg_collect_time += eplased_time
                debug_print("COLLECT", f"Collecting time over limit. t={eplased_time}", "WARNING")
            else:
                avg_collect_time += 1 / save_freq
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
            debug_print("COLLECT", f"Total frame: {collect_num}", "INFO")
            
            avg_collect_time = avg_collect_time / collect_num
            extra_info["collect_num"] = collect_num
            extra_info["avg_time_interval"] = avg_collect_time
        self.robot.collector.add_extra_cfg_info(extra_info)