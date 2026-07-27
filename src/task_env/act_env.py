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
import threading
import rerun as rr

import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, policy_cfg: Dict[str, Any]):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(policy_cfg)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = policy_cfg['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat
        

def load_policy(policy_cfg: Dict[str, Any]):
    """加载模型 checkoint 和归一化参数"""
    ckpt_path = os.path.join(policy_cfg['ckpt_dir'], 'policy_best.ckpt')
    stats_path = os.path.join(policy_cfg['ckpt_dir'], 'dataset_stats.pkl')

    camera_names = policy_cfg['camera_names']

    override_policy_cfg = {
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

    policy = ACTPolicy(override_policy_cfg)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 预处理/后处理函数
    pre_process = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    return policy, pre_process, post_process

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

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
        self.enable_rerun = base_cfg.get("enable_rerun", False)

        self.policy_cfg = base_cfg.get("act_policy", {})
        debug_print(self.name, f"Policy config: {self.policy_cfg}", "INFO")
        self.policy, self.pre_process, self.post_process = None, None, None
        self.chunk_size = self.policy_cfg['chunk_size']
        self.hidden_dim = self.policy_cfg['hidden_dim']
        self.dim_feedforward = self.policy_cfg['dim_feedforward']
        self.temporal_agg = self.policy_cfg['temporal_agg']
        self.max_timesteps = self.policy_cfg['max_timesteps']
        self.camera_names = self.policy_cfg['camera_names']  # 默认相机列表

        # 共享变量（线程安全）
        self.inference_thread = None
        self.stop_thread = False
        self.lock = threading.Lock()
        self.latest_action = None  # 最新的推理结果
        self.inference_fps = 10
        self.control_fps = 30

    def set_up(self):
        super().set_up()
        self.policy, self.pre_process, self.post_process = load_policy(self.policy_cfg)

        self.stop_thread = False
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        if self.enable_rerun:
            rr.init("collection", spawn=False)
            server_url = rr.serve_grpc(
                grpc_port=9876,
                server_memory_limit="1GiB",
                newest_first=False,
                cors_allow_origin=["*"]
            )
            debug_print("COLLECT", f"Rerun gRPC 服务器已启动: {server_url}", "INFO")
            rr.serve_web_viewer(
                web_port=9090,  # Web 界面端口
                open_browser=False,  # 自动打开浏览器
                # connect_to=server_url  # 连接到 gRPC 服务器
            )
    
        debug_print(self.name, "Environment setup complete.", "INFO")

    def trans_obs_to_act_input(self, data):
        controller_data, sensor_data = data[0], data[1]
        # --- build qpos ----------------------------------------------------
        qpos = np.zeros(14, dtype=np.float32)
        for arm in _DEFAULT_DUAL_ARM_MAPPING:
            ctrl_name = arm["controller_name"]

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
            # if ctrl_name == "slave_left_arm":
            #     move_data["arm"][ctrl_name] = {
            #         "joint": np.zeros(6, dtype=np.float32),
            #         "gripper": 1.0,
            #     }

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
        debug_print(self.name, "Inference thread started", "INFO")
        max_timesteps = self.max_timesteps
        inference_interval = 1 / self.inference_fps
        # 如果启用时序平滑，则每步只查询一次策略，否则每 chunk_size 步查询一次
        query_frequency = self.chunk_size if not self.temporal_agg else 1
        state_dim = 14
        if self.temporal_agg:
            all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.chunk_size, state_dim]
            ).cuda()
        timestep = 0

        try:
            with torch.inference_mode():
                while not self.stop_thread:
                    loop_start = time.monotonic()
                    tick_1 = 0
                    
                    rr.set_time("frame", sequence=timestep)

                    obs = self.robot.get_obs()
                    self.robot.visualize()
                    act_obs = self.trans_obs_to_act_input(obs)

                    if act_obs is not None:
                        curr_qpos = torch.from_numpy(self.pre_process(act_obs["qpos"])).float().cuda().unsqueeze(0)
                        curr_image = self.build_image_tensor(act_obs)
                        tick_1 = time.monotonic() - loop_start
                        if timestep % query_frequency == 0:
                            all_actions = self.policy(curr_qpos, curr_image)
                        tick_2 = time.monotonic() - loop_start - tick_1
                        if self.temporal_agg:
                            f = 0
                            # 时序平滑：对同一时刻的多个预测做指数加权
                            all_time_actions[[timestep], timestep:timestep + self.chunk_size] = all_actions
                            actions_for_step = all_time_actions[:, timestep + f]
                            populated = torch.all(actions_for_step != 0, axis=1)
                            actions_for_step = actions_for_step[populated]
                            k = 0.01
                            weights = np.exp(-k * np.arange(len(actions_for_step)))
                            weights = weights / weights.sum()
                            weights = torch.from_numpy(weights).cuda().unsqueeze(1)
                            raw_action = (actions_for_step * weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, timestep % query_frequency]

                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        target_qpos = self.post_process(raw_action)

                        # 更新共享动作（加锁）
                        with self.lock:
                            self.latest_action = target_qpos.copy()

                        timestep += 1
                        if timestep >= max_timesteps:
                            self.stop_thread = True

                    # 控制推理频率
                    elapsed = time.monotonic() - loop_start
                    sleep_time = inference_interval - elapsed
                    debug_print(self.name, f"Trans Data completed in {tick_1:.4f} seconds.", "WARNING")
                    debug_print(self.name, f"Query policy time: {tick_2:.4f} seconds.", "INFO")
                    debug_print(self.name, f"Step {timestep}/{max_timesteps} completed in {elapsed:.4f} seconds.", "INFO")
                    if sleep_time > 0:
                        time.sleep(sleep_time)


        except Exception as e:
            debug_print(self.name, f"Inference error: {e}", "ERROR")

    def run_deployment(self):
        self.robot.reset()

        control_interval = 1 / self.control_fps
        target_qpos = None
        debug_print(self.name, "Deployment started. Waiting for user input to continue.", "INFO")
        while True:
            ch = read_key()
            if ch == KEY_DICT["START"]:
                debug_print(self.name, "Start to execute.", "INFO")
                break
            elif ch == KEY_DICT["QUIT"]:
                debug_print(self.name, "User requested exit.", "INFO")
                self.robot.reset()
                return
        policy_qpos = None
        try:
            while not self.stop_thread:
                loop_start = time.monotonic()
                
                with self.lock:
                    if self.latest_action is not None:
                        policy_qpos = self.latest_action.copy()

                if policy_qpos is None:
                    debug_print(self.name, f"No policy_qpos available", "WARNING")
                else:
                    # debug_print(self.name, f"Policy: {np.round(policy_qpos[7:14], 4)}", "INFO")
                    if target_qpos is None:
                        target_qpos = np.array(policy_qpos)
                    else:
                        rate = 0.3
                        target_qpos = np.array(policy_qpos) * rate + target_qpos * (1 - rate)
                        debug_print(self.name, f"Target: {np.round(target_qpos[7:14], 4)}", "INFO")
                        target_qpos[13] = policy_qpos[13] 
                    move_data = self.trans_act_action_to_move_data(target_qpos)
                    self.robot.move(move_data)

                elapsed = time.monotonic() - loop_start
                if elapsed < control_interval:
                    time.sleep(control_interval - elapsed)
                else:
                    debug_print(self.name, "Move command execute over time limit", "WARNING")

        except KeyboardInterrupt:
            debug_print(self.name, "User interrupted. Exiting.", "INFO")
        finally:
            self.robot.reset()
            self.env_finish()
            debug_print(self.name, "Deployment finished. Robot reset.", "INFO")