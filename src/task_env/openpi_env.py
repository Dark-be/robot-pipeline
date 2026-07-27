import numpy as np
import time
import threading
import queue
from typing import Any, Dict, List, Optional
import logging

from robot.utils.base.data_handler import debug_print, read_key, KEY_DICT
from .base_env import BaseEnv

# OpenPI 客户端导入
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import image_tools

class OpenPIPolicy:
    """
    OpenPI 策略客户端
    直接与本地推理服务器通信
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.server_url = config.get("server_url", "ws://127.0.0.1:8000")
        self.api_key = config.get("api_key", None)
        self.resize_size = config.get("resize_size", 224)
        self.state_dim = config.get("state_dim", 14)
        self.prompt = config.get("prompt", "perform the task")
        
        # 连接服务器
        self._connect_server()
        
    def _connect_server(self):
        """连接到 OpenPI 服务器"""
        host, port = self._parse_url(self.server_url)
        self.client = _websocket_client_policy.WebsocketClientPolicy(
            host=host, 
            port=port, 
            api_key=self.api_key
        )
        self.server_metadata = self.client.get_server_metadata()
        debug_print(self.name, f"Connected to OpenPI server: {self.server_metadata}", "INFO")
    
    def _parse_url(self, url: str) -> tuple:
        """解析服务器URL"""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme:
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 8000
            return host, port
        return url, 8000
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """预处理图像为模型输入格式"""
        return image_tools.resize_with_pad(
            img.astype(np.uint8, copy=False),
            self.resize_size,
            self.resize_size
        )
    
    def infer(self, qpos: np.ndarray, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        执行推理，返回动作序列
        """
        # 预处理图像
        processed_images = {
            name: self.preprocess_image(img)
            for name, img in images.items()
        }
        
        # 构建观测
        observation = {
            "images": processed_images,
            "image_masks": {
                name: True for name in images.keys()
            },
            "state": qpos.astype(np.float32),
            "prompt": self.prompt,
        }
        
        # 发送推理请求
        response = self.client.infer(observation)
        
        # 提取动作
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise RuntimeError(f"Expected action chunk [T, 14], got shape {actions.shape}")
        
        # 只取前14维
        actions = actions[:, :self.state_dim]
        
        return actions
    
    def close(self):
        """关闭连接"""
        self.client = None


class OpenPIEnv(BaseEnv):
    """
    OpenPI 策略环境
    简化版，直接使用 piper_infer.py 的逻辑
    """
    
    def __init__(self, base_cfg):
        super().__init__(base_cfg=base_cfg)
        self.name = "OpenPIEnv"
        
        # 策略配置
        self.policy_config = base_cfg.get("openpi_policy", {})
        debug_print("OpenPIEnv", f"Policy config: {self.policy_config}", "INFO")
        
        # 策略参数
        self.chunk_size = self.policy_config.get("chunk_size", 5)
        self.replan_steps = self.policy_config.get("replan_steps", 5)
        self.camera_names = self.policy_config.get("camera_names", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        
        # 策略对象
        self.policy = None
        
        # 线程控制
        self.inference_thread = None
        self.stop_thread = False
        
        # 共享变量（线程安全）
        self.lock = threading.Lock()
        self.latest_obs = None  # 最新观测
        self.action_queue = queue.Queue()  # 动作队列
        
        # 控制变量
        self.target_qpos = None
        self.step_count = 0
        
    def set_up(self):
        """初始化环境"""
        super().set_up()
        
        # 初始化策略
        self.policy = OpenPIPolicy(self.policy_config)
        
        # 启动推理线程
        self.stop_thread = False
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        
        debug_print("OpenPIEnv", "Environment setup complete.", "INFO")
    
    def trans_obs_2_act(self, data) -> Optional[Dict[str, Any]]:
        """
        将机器人观测转换为 OpenPI 格式
        """
        controller_data, sensor_data = data[0], data[1]
        
        # --- 构建 qpos (14维) ---
        qpos = np.zeros(14, dtype=np.float32)
        
        # 左臂: 索引 0-5 关节, 6 夹爪
        left_ctrl = controller_data.get("slave_left_arm", {})
        left_joint = np.asarray(left_ctrl.get("joint", []), dtype=np.float32).ravel()
        left_gripper = np.asarray(left_ctrl.get("gripper", [0.0]), dtype=np.float32).ravel()
        
        for i in range(min(6, len(left_joint))):
            qpos[i] = left_joint[i]
        if len(left_gripper) > 0:
            qpos[6] = left_gripper[0]
        
        # 右臂: 索引 7-12 关节, 13 夹爪
        right_ctrl = controller_data.get("slave_right_arm", {})
        right_joint = np.asarray(right_ctrl.get("joint", []), dtype=np.float32).ravel()
        right_gripper = np.asarray(right_ctrl.get("gripper", [0.0]), dtype=np.float32).ravel()
        
        for i in range(min(6, len(right_joint))):
            qpos[7 + i] = right_joint[i]
        if len(right_gripper) > 0:
            qpos[13] = right_gripper[0]
        
        # --- 构建图像 ---
        images: Dict[str, np.ndarray] = {}
        for cam_name in self.camera_names:
            if cam_name not in sensor_data:
                debug_print(self.name, f"Camera '{cam_name}' not found", "WARNING")
                continue
            img_raw = sensor_data[cam_name].get("color")
            if img_raw is None:
                debug_print(self.name, f"No color image for '{cam_name}'", "WARNING")
                continue
            images[cam_name] = np.asarray(img_raw, dtype=np.uint8)
        
        # 检查是否有有效图像
        if not images:
            return None
        
        return {"qpos": qpos, "images": images}
    
    def trans_act_action_to_move_data(self, action: np.ndarray) -> Dict[str, Any]:
        """
        将动作数组转换为 Robot.move() 命令
        """
        action = np.asarray(action, dtype=np.float32)
        
        move_data: Dict[str, Dict[str, Dict[str, Any]]] = {"arm": {}}
        
        # 左臂: 索引 0-6
        left_joint = action[0:6]
        left_gripper = np.clip(action[6], 0.0, 1.0)
        
        move_data["arm"]["slave_left_arm"] = {
            "joint": left_joint,
            "gripper": left_gripper,
        }
        
        # 右臂: 索引 7-13
        right_joint = action[7:13]
        right_gripper = np.clip(action[13], 0.0, 1.0)
        
        move_data["arm"]["slave_right_arm"] = {
            "joint": right_joint,
            "gripper": right_gripper,
        }
        
        return move_data
    
    def inference_loop(self):
        """
        推理线程主循环
        类似于 piper_infer.py 的 main 循环
        """
        debug_print("OpenPIEnv", "Inference thread started", "INFO")
        
        inference_interval = 1.0 / self.policy_config.get("inference_fps", 10.0)
        max_timesteps = self.policy_config.get("max_timesteps", 1000)
        
        step = 0
        action_buffer = []  # 动作缓存
        
        try:
            while not self.stop_thread and step < max_timesteps:
                loop_start = time.monotonic()
                
                # 获取最新观测（加锁）
                obs_copy = None
                with self.lock:
                    if self.latest_obs is not None:
                        obs_copy = {
                            'qpos': self.latest_obs['qpos'].copy(),
                            'images': {k: v.copy() for k, v in self.latest_obs['images'].items()}
                        }
                
                if obs_copy is not None:
                    # 如果动作缓存为空，请求新的动作块
                    if not action_buffer:
                        try:
                            # 执行推理
                            actions = self.policy.infer(
                                obs_copy['qpos'],
                                obs_copy['images']
                            )
                            
                            # 取前 replan_steps 个动作
                            if len(actions) < self.replan_steps:
                                debug_print(self.name, 
                                          f"Policy returned {len(actions)} actions, "
                                          f"expected {self.replan_steps}", "WARNING")
                            
                            action_buffer = list(actions[:self.replan_steps])
                            debug_print("OpenPIEnv", 
                                      f"Got {len(action_buffer)} new actions", "INFO")
                            
                        except Exception as e:
                            debug_print("OpenPIEnv", f"Inference error: {e}", "ERROR")
                            time.sleep(0.1)
                            continue
                    
                    # 从缓存中取出一个动作
                    if action_buffer:
                        action = action_buffer.pop(0)
                        self.action_queue.put(action)
                        step += 1
                
                # 控制推理频率
                elapsed = time.monotonic() - loop_start
                sleep_time = inference_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            debug_print("OpenPIEnv", f"Inference thread error: {e}", "ERROR")
        finally:
            debug_print("OpenPIEnv", "Inference thread stopped", "INFO")
    
    def run_deployment(self):
        """
        主部署循环
        类似于 piper_infer.py 的执行循环
        """
        # 重置机器人
        self.robot.reset()
        
        # 等待用户输入开始
        allow_step = 0
        fps = self.policy_config.get("control_fps", 15.0)
        target_qpos = None
        
        print("[OpenPIEnv] Press ENTER to start deployment, Q to quit")
        while allow_step <= 0:
            ch = read_key()
            if ch == KEY_DICT["CONTINUE"] or ch == KEY_DICT["ENTER"]:
                allow_step = 100
                print("[OpenPIEnv] Starting deployment...")
                break
            elif ch == KEY_DICT["QUIT"]:
                print("[OpenPIEnv] User requested exit.")
                self.robot.reset()
                return
        
        try:
            while not self.stop_thread:
                loop_start = time.monotonic()
                
                # 获取观测并更新
                data = self.robot.get_obs()
                act_obs = self.trans_obs_2_act(data)
                if act_obs is not None:
                    with self.lock:
                        self.latest_obs = act_obs
                
                # 从队列获取动作
                try:
                    policy_action = self.action_queue.get_nowait()
                except queue.Empty:
                    policy_action = None
                
                if policy_action is not None:
                    target_qpos = np.array(policy_action)
                    debug_print(self.name, 
                              f"Action: left_gripper={target_qpos[6]:.3f}, "
                              f"right_gripper={target_qpos[13]:.3f}", 
                              "INFO")
                
                # 执行动作
                if target_qpos is not None:
                    move_data = self.trans_act_action_to_move_data(target_qpos)
                    self.robot.move(move_data)
                
                # 控制频率
                elapsed = time.monotonic() - loop_start
                if elapsed < 1.0 / fps:
                    time.sleep(1.0 / fps - elapsed)
                    
        except KeyboardInterrupt:
            debug_print("OpenPIEnv", "User interrupted. Exiting.", "INFO")
        finally:
            # 清理
            self.stop_thread = True
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=2.0)
            self.robot.reset()
            if self.policy:
                self.policy.close()
            self.env_finish()
            debug_print("OpenPIEnv", "Deployment finished. Robot reset.", "INFO")