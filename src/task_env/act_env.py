import torch
import numpy as np
import os
import pickle
import time
from typing import Any, Dict, List
from einops import rearrange
# import torchvision.transforms as transforms
from robot.utils.base.data_handler import debug_print, read_key, KEY_DICT
from .base_env import BaseEnv
from .policy import ACTPolicy
import threading
from collections import deque

def load_policy(policy_cfg):
    """加载模型 checkoint 和归一化参数"""
    ckpt_path = os.path.join(policy_cfg['ckpt_dir'], 'policy_best.ckpt')
    stats_path = os.path.join(policy_cfg['ckpt_dir'], 'dataset_stats.pkl')

    camera_names = policy_cfg['camera_names']

    policy_config = {
        'lr': 1e-5,
        'num_queries': policy_cfg['chunk_size'],
        'kl_weight': 10,
        'hidden_dim': policy_cfg['hidden_dim'],
        'dim_feedforward': policy_cfg['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
    }

    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 预处理/后处理函数
    pre_process = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    return policy, pre_process, post_process

_DEFAULT_DUAL_ARM_MAPPING: List[Dict[str, Any]] = [
    {
        "controller_name": "slave_left_arm",
        "joint_indices": [0, 1, 2, 3, 4, 5],
        "gripper_index": 6,
    },
    {
        "controller_name": "slave_right_arm",
        "joint_indices": [7, 8, 9, 10, 11, 12],
        "gripper_index": 13,
    },
]

class ACTEnv(BaseEnv):
    def __init__(self, base_cfg):
        super().__init__(base_cfg=base_cfg)
        self.name = "ACTEnv"

        self.policy_config = base_cfg.get("act_policy", {})
        debug_print("ACTEnv", f"Policy config: {self.policy_config}", "INFO")
        self.policy, self.pre_process, self.post_process = None, None, None
        self.chunk_size = self.policy_config['chunk_size']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.dim_feedforward = self.policy_config['dim_feedforward']
        self.temporal_agg = self.policy_config['temporal_agg']
        self.max_timesteps = self.policy_config['max_timesteps']
        self.camera_names = ["cam_head", "cam_right_wrist"]  # 默认相机列表

        self.inference_thread = None
        self.stop_thread = False
        
        # 共享变量（线程安全）
        self.lock = threading.Lock()
        self.latest_obs = None  # 最新的观测数据
        self.latest_action = None  # 最新的推理结果
        self.action_ready = False  # 是否有新的动作可用
        self.inference_interval = 0.2  # 推理间隔（20Hz）
        self.control_fps = 20

    def set_up(self):
        super().set_up()
        self.policy, self.pre_process, self.post_process = load_policy(self.policy_config)

        self.stop_thread = False
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
    
    debug_print("ACTEnv", "Environment setup complete.", "INFO")

    def trans_obs_2_act(self, data):
        controller_data, sensor_data = data[0], data[1]
        # --- build qpos ----------------------------------------------------
        qpos = np.zeros(14, dtype=np.float32)
        for arm in _DEFAULT_DUAL_ARM_MAPPING:
            ctrl_name = arm["controller_name"]
            if ctrl_name == "slave_left_arm":
                continue
            ctrl_data = controller_data.get(ctrl_name, {})

            joint = np.asarray(ctrl_data.get("joint", []), dtype=np.float32).ravel()
            gripper = np.asarray(ctrl_data.get("gripper", [0.0]), dtype=np.float32).ravel()

            joint_indices = arm.get("joint_indices", [])
            gripper_index = arm.get("gripper_index")

            # 填充关节值
            for idx, val in zip(joint_indices, joint):
                if idx < 14:  # 安全保护
                    qpos[idx] = float(val)
        
            # 填充夹爪值
            if gripper_index is not None and gripper_index < 14:
                qpos[gripper_index] = float(gripper[0]) if len(gripper) > 0 else 0.0


        # --- build images --------------------------------------------------
        images: Dict[str, np.ndarray] = {}
        for cam_name in self.camera_names:
            if cam_name not in sensor_data:
                raise KeyError(
                    f"Camera '{cam_name}' not found in sensor_data. "
                    f"Available: {list(sensor_data.keys())}"
                )
            img_raw = sensor_data[cam_name].get("color")
            if img_raw is None:
                return None
            else:
                images[cam_name] = np.asarray(img_raw, dtype=np.uint8)

        return {"qpos": qpos, "images": images}

    def trans_act_action_to_move_data(self, action: np.ndarray) -> dict:
        """Convert flat ACT action array to Robot.move() dict.

        ACT format:
            action: np.ndarray (state_dim,)  -- flat joint + gripper values

        Robot format:
            {"arm": {"arm_name": {"joint": array, "gripper": scalar}}}
        """
        move_data: Dict[str, Dict[str, Dict[str, Any]]] = {"arm": {}}
        for arm in _DEFAULT_DUAL_ARM_MAPPING:
            ctrl_name = arm["controller_name"]
            joint_vals = action[arm["joint_indices"]].astype(np.float32)
            gi = arm.get("gripper_index")
            gripper_val = float(action[gi]) if gi is not None else 1.0

            move_data["arm"][ctrl_name] = {
                "joint": joint_vals,
                "gripper": np.clip(gripper_val, 0.0, 1.0),
            }
            if ctrl_name == "slave_left_arm":
                move_data["arm"][ctrl_name] = {
                    "joint": np.zeros(6, dtype=np.float32),
                    "gripper": 1.0,
                }

        return move_data

    def build_image_tensor(self, obs):
        """从 observation 中提取多相机图像并转为模型输入格式"""
        curr_images = []
        for cam_name in self.camera_names:
            img = obs['images'][cam_name]  # (H, W, 3) uint8
            img = rearrange(img, 'h w c -> c h w')
            curr_images.append(img)
        image = np.stack(curr_images, axis=0)
        image = torch.from_numpy(image / 255.0).float().cuda().unsqueeze(0)
        return image
    
    def inference_loop(self):
        debug_print("ACTEnv", "Inference thread started", "INFO")
        max_timesteps = self.max_timesteps
        # 如果启用时序平滑，则每步只查询一次策略，否则每 chunk_size 步查询一次
        query_frequency = self.chunk_size if not self.temporal_agg else 1
        state_dim = 14
        if self.temporal_agg:
            all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.chunk_size, state_dim]
            ).cuda()
        t = 0
        try:
            with torch.inference_mode():
                while not self.stop_thread:
                    start_time = time.monotonic()
                    obs_copy = None
                    with self.lock:
                        if self.latest_obs is not None:
                            obs_copy = {
                                'qpos': self.latest_obs['qpos'].copy(),
                                'images': {k: v.copy() for k, v in self.latest_obs['images'].items()}
                            }
                    if obs_copy is not None:
                        curr_qpos = torch.from_numpy(self.pre_process(obs_copy["qpos"])).float().cuda().unsqueeze(0)
                        curr_image = self.build_image_tensor(obs_copy)

                        if t % query_frequency == 0:
                            all_actions = self.policy(curr_qpos, curr_image)

                        if self.temporal_agg:
                            # 时序平滑：对同一时刻的多个预测做指数加权
                            all_time_actions[[t], t:t + self.chunk_size] = all_actions
                            actions_for_step = all_time_actions[:, t]
                            populated = torch.all(actions_for_step != 0, axis=1)
                            actions_for_step = actions_for_step[populated]
                            k = 0.01
                            weights = np.exp(-k * np.arange(len(actions_for_step)))
                            weights = weights / weights.sum()
                            weights = torch.from_numpy(weights).cuda().unsqueeze(1)
                            raw_action = (actions_for_step * weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]

                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        target_qpos = self.post_process(raw_action)

                        # 更新共享动作（加锁）
                        with self.lock:
                            self.latest_action = target_qpos.copy()
                            self.action_ready = True

                        t += 1
                        if t >= max_timesteps:
                            self.stop_thread = True

                    # 控制推理频率
                    elapsed = time.monotonic() - start_time
                    sleep_time = self.inference_interval - elapsed
                    debug_print("ACTEnv", f"Step {t}/{max_timesteps} completed in {elapsed:.4f} seconds.", "INFO")
                    if sleep_time > 0:
                        time.sleep(sleep_time)


        except Exception as e:
            debug_print("ACTEnv", f"Inference error: {e}", "ERROR")


    def run_deployment(self):
        self.robot.reset()

        allow_step = 0
        fps = self.control_fps
        target_qpos = None
        print("ok")
        while allow_step <= 0:
            ch = read_key()
            if ch == KEY_DICT["CONTINUE"]:
                allow_step = 100
                print("[PolicyEnv] Continuing to next step.")
                break
            elif ch == KEY_DICT["QUIT"]:
                print("[PolicyEnv] User requested exit.")
                self.robot.reset()
                return
        try:
            while not self.stop_thread:
                start_time = time.monotonic()

                # 传入推理
                data = self.robot.get_obs()
                act_obs = self.trans_obs_2_act(data)
                if act_obs is not None:
                    with self.lock:
                        self.latest_obs = act_obs
                
                policy_qpos = None
                with self.lock:
                    if self.latest_action is not None:
                        policy_qpos = self.latest_action.copy()

                if policy_qpos is None:
                    debug_print(self.name, f"No target", "WARNING")
                else:
                    debug_print(self.name, f"Policy: {policy_qpos[7:14]}", "INFO")
                    if target_qpos is None:
                        target_qpos = np.array(policy_qpos)
                    else:
                        rate = 0.8
                        target_qpos = np.array(policy_qpos) * rate + target_qpos * (1 - rate)
                        if target_qpos[13] < 0.5:
                            target_qpos[13] = 0.05
                    move_data = self.trans_act_action_to_move_data(target_qpos)
                    self.robot.move(move_data)

                elapsed_time = time.monotonic() - start_time
                if elapsed_time < 1 / fps:
                    time.sleep(1 / fps - elapsed_time)
                else:
                    debug_print(self.name, "Overlimit", "WARNING")

        except KeyboardInterrupt:
            debug_print("ACTEnv", "User interrupted. Exiting.", "INFO")
        finally:
            self.robot.reset()
            self.env_finish()
            debug_print("ACTEnv", "Deployment finished. Robot reset.", "INFO")